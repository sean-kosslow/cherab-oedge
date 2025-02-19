
# Copyright 2014-2017 United Kingdom Atomic Energy Authority
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

# External imports
from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt
import numpy as np
from raysect.optical import World, Point3D, Vector3D, Spectrum
from raysect.optical.material import AbsorbingSurface
from sal.client import SALClient

# Internal imports
from cherab.core.utility import PhotonToJ
from cherab.core.atomic import Line, deuterium
from cherab.core.model import ExcitationLine, RecombinationLine
from cherab.openadas import OpenADAS
from cherab.oedge import load_edge2d_from_tranfile
from cherab.jet.machine import import_jet_mesh
from cherab.jet.spectroscopy.ks3 import load_ks3_inner_array, load_ks3_outer_array, array_polychromator


sal = SALClient('https://sal.jet.uk')


def plot_dalpha_emission(mesh, plasma, ks3_inner_array, ks3_outer_array):
    me = mesh.mesh_extent
    rl, ru = (me['minr'], me['maxr'])
    zl, zu = (me['minz'], me['maxz'])
    nr = 500
    nz = 1000
    rsamp = np.linspace(rl, ru, nr)
    zsamp = np.linspace(zl, zu, nz)
    emission = np.zeros((nz, nr))
    direction = Vector3D(0, 0, 1)
    for i, x in enumerate(rsamp):
        for j, z in enumerate(zsamp):
            point = Point3D(x, 0, z)
            spectrum = Spectrum(655., 657., 1)
            for model in plasma.models:
                emission[j, i] += model.emission(point, direction, spectrum).total()
    # plot emissivity
    fig, ax = plt.subplots(figsize=(6., 7.), constrained_layout=True)
    linthresh = np.percentile(np.unique(emission), 1)
    norm = SymLogNorm(linthresh=linthresh)
    image = ax.imshow(emission, extent=[rl, ru, zl, zu], origin='lower', norm=norm)
    fig.colorbar(image, label='W m-3 sr-1', aspect=40)
    ax.set_xlim(rl, ru)
    ax.set_ylim(zl, zu)
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')

    # plot lines of sight
    length = 5.5
    for (los_group, color) in ((ks3_inner_array, '0.5'), (ks3_outer_array, '1.0')):
        for sight_line in los_group.sight_lines:
            origin = sight_line.origin
            direction = sight_line.direction
            radius = sight_line.radius
            angle = np.deg2rad(sight_line.acceptance_angle)
            end = origin + length * direction
            radius_end = radius + np.tan(angle) * length
            ro = np.sqrt(origin.x**2 + origin.y**2)
            zo = origin.z
            re = np.sqrt(end.x**2 + end.y**2)
            ze = end.z
            theta = 0.5 * np.pi - np.arctan2(zo - ze, ro - re)
            rr = (ro + radius * np.cos(theta), re + radius_end * np.cos(theta))
            rl = (ro - radius * np.cos(theta), re - radius_end * np.cos(theta))
            zr = (zo + radius * np.sin(theta), ze + radius_end * np.sin(theta))
            zl = (zo - radius * np.sin(theta), ze - radius_end * np.sin(theta))
            ax.plot(rr, zr, color=color, lw=0.75)
            ax.plot(rl, zl, color=color, lw=0.75)
            ax.plot((ro, re), (zo, ze), ls='--', color=color, lw=0.75)

    return ax


def load_ks3_pmt_array_data(pulse, time, userid, signal, sequence=0, window=0.05):
    data = sal.get('/pulse/{}/ppf/signal/{}/edg8/{}:{}'.format(pulse, userid, signal, sequence))
    t = data.dimensions[0].data
    it_min = np.abs(t - time + 0.5 * window).argmin()  # closest time moment
    it_max = np.abs(t - time - 0.5 * window).argmin()  # closest time moment

    return data.data[it_min:it_max].mean(0) * 1.e4  # cm-2 --> m-2


# ----Creating plasma from EDGE2D simulation---- #

user = 'pheliste'
edge2d_case = '81472/jul1716/seq#4'
tranfile = '/home/{}/cmg/catalog/edge2d/jet/{}/tran'.format(user, edge2d_case)
pulse = 94767
time = 51.

sim = load_edge2d_from_tranfile(tranfile)
plasma = sim.create_plasma()
plasma.atomic_data = OpenADAS(permit_extrapolation=True)
plasma.integrator.step = 0.005

# ----Adding Gaussian D-alpha line shape model---- #

d_alpha = Line(deuterium, 0, (3, 2))
plasma.models = [ExcitationLine(d_alpha), RecombinationLine(d_alpha)]
wavelength = plasma.atomic_data.wavelength(deuterium, 0, (3, 2))

# ----Loading diagnostics---- #

ks3_inner = load_ks3_inner_array(pulse, instruments=[array_polychromator])
ks3_outer = load_ks3_outer_array(pulse, instruments=[array_polychromator])
ks3_inner.pixel_samples = 5000
ks3_outer.pixel_samples = 5000

# ----Plotting H-alpha emissivity and diagnostic geometry---- #

plt.ion()
ax = plot_dalpha_emission(sim.mesh, plasma, ks3_inner, ks3_outer)
ax.set_title('D-alpha emissivity\nEDGE2D #{}/{} + ADAS'.format(user, edge2d_case))

# ----Observing with reflections---- #

world = World()
plasma.parent = world

# loading wall mesh
jet_mesh = import_jet_mesh(world)

pipeline_name = 'array_polychromator: D alpha'
radiance_refl_wall = {}
for los_group in (ks3_inner, ks3_outer):
    los_group.parent = world
    los_group.observe()
    observed_radiance = np.array([sightline.get_pipeline(pipeline_name).value.mean for sightline in los_group.sight_lines])
    radiance_refl_wall[los_group] = PhotonToJ.inv(observed_radiance, wavelength)

# ----Observing without reflections---- #

# changing wall material to AbsorbingSurface
absorbing_surface = AbsorbingSurface()
for mesh_component in jet_mesh:
    mesh_component.material = absorbing_surface

radiance_abs_wall = {}
for los_group in (ks3_inner, ks3_outer):
    los_group.observe()
    observed_radiance = np.array([sightline.get_pipeline(pipeline_name).value.mean for sightline in los_group.sight_lines])
    radiance_abs_wall[los_group] = PhotonToJ.inv(observed_radiance, wavelength)

# ----Reading the experimental values---- #

radiance_exp = {}
for (los_group, signal_name) in ((ks3_inner, 'dai'), (ks3_outer, 'dao')):
    radiance_exp[los_group] = load_ks3_pmt_array_data(pulse, time, 'jetppf', signal_name)

# ----Plotting the results---- #

for los_group in (ks3_inner, ks3_outer):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(np.arange(1, 11), radiance_exp[los_group], color='k', ls='none', marker='s', mfc='none',
            label='Pulse: {}, time: {} s'.format(pulse, time))
    ax.plot(np.arange(1, 11), radiance_refl_wall[los_group], ls='none', marker='x', label='EDGE2D with reflections')
    ax.plot(np.arange(1, 11), radiance_abs_wall[los_group], ls='none', marker='o', mfc='none', label='EDGE2D without reflections')
    ax.legend(loc=1, frameon=False)
    ax.set_xlabel('Line of sight index')
    ax.set_ylabel('D-alpha Radiance, photon s-1 sr-1 m-2')
    ax.set_title('{}, {}/{}'.format(los_group.name, user, edge2d_case))

plt.ioff()
plt.show()
