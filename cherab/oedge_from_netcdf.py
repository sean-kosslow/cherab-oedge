""" Copyright 2016-2018 Euratom
Copyright 2016-2018 United Kingdom Atomic Energy Authority
Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas

Licensed under the EUPL, Version 1.1 or as soon they will be approved by the
European Commission - subsequent versions of the EUPL (the "Licence");
You may not use this work except in compliance with the Licence.
You may obtain a copy of the Licence at:

https://joinup.ec.europa.eu/software/page/eupl5

Unless required by applicable law or agreed to in writing, software distributed
under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.

See the Licence for the specific language governing permissions and limitations
under the Licence.

Development of a script for reading in oegde plasmas from the processed data
Sean Kosslow 6/16/2022

This script is written to run on my laptop, you will need to change file paths to fit the 
machine you will be running it on -SRK"""

import os
import time
from datetime import datetime
## External imports
from operator import matmul

import hickle as hkl
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.constants import atomic_mass, electron_mass
from scipy.interpolate import interp2d

tic = time.perf_counter()
print(datetime.now())
# os.environ['OPENBLAS_NUM_THREADS']='8'

import os

## CHERAB imports
from cherab.core import Line, Maxwellian, Plasma, Species, elements
from cherab.core.atomic.elements import deuterium, helium, tungsten
from cherab.core.math import (AxisymmetricMapper, ConstantVector3D,
                              VectorAxisymmetricMapper)
from cherab.core.model import (Bremsstrahlung, ExcitationLine, GaussianLine,
                               RecombinationLine, StarkBroadenedLine)
## CHERAB-OEDGE imports
from cherab.edge2d.edge2d_plasma import Edge2DSimulation
from cherab.edge2d.mesh_geometry import Edge2DMesh
from cherab.openadas import OpenADAS
from cherab.tools.observers import FibreOpticGroup
from cherab.tools.observers.spectroscopy.fibreoptic import SpectroscopicFibreOptic
## Raysect imports
from raysect.core import MulticoreEngine, SerialEngine
from raysect.core.math import Vector3D
from raysect.core.math.function.float.function3d.base import Function3D
from raysect.core.ray import Ray
from raysect.optical import (ConstantSF, Point3D, Ray, World, rotate_basis,
                             rotate_x, rotate_y, rotate_z, translate)
from raysect.optical.library.metal import RoughTungsten
from raysect.optical.material import AbsorbingSurface, Lambert
from raysect.optical.material.debug import Light
from raysect.optical.observer import (CCDArray, FibreOptic, PinholeCamera,
                                      PowerPipeline2D, RadiancePipeline0D,
                                      RGBPipeline2D, SpectralPowerPipeline0D,
                                      SpectralRadiancePipeline0D, VectorCamera)
from raysect.primitive import Cylinder, Sphere
from raysect.primitive.mesh.stl import import_stl

# from cherab.west.machine.fromtofu import import_west_mesh_from_tofu 

# from cherab.oedge.edge2d_plasma import OEDGESimulation

os.environ['OPENBLAS_NUM_THREADS'] = '1'

world = World()

Q = 1.602e-19 #electron charge

## select atomic data source (This will get supplemented with Curt Johnson's W hdf5 file later)
adas = OpenADAS(permit_extrapolation=True)

## This geometry is for visualization purposes ONLY. It is not the 3D CAD that is used as the mesh for ray tracing
Rwall = pd.read_csv('~/projects/Rwall_antenna_back.txt')
Zwall = pd.read_csv('~/projects/Zwall.txt')
R = Rwall.to_numpy()
Z = Zwall.to_numpy()

## Import WEST geometry from stl file (Thanks Romain) and specify material
## Simple CAD version
# westmesh = import_stl('/home/skosslow/projects/WESTgeom3D.stl', scaling=0.001, mode='auto', parent = world)
# roughness = 0.1
# westmesh.material = RoughTungsten(roughness) #Added 3/8/2022 to replace stand-in reflector
# # westmesh.material = AbsorbingSurface() # Can be turned on in place of the tungsten to debug
# westmesh.transform = translate(0,0,0)
# westmesh.transform = rotate_basis(Vector3D(1,0,0),Vector3D(0,0,1))
# # westmesh.transform = rotate_y(90)

## High detail CAD version
print('Import WEST CAD Geometry from .stl file')
westmesh = []
stl_list = ['Antenne_1_PhantomMesh_024.stl',
            'Antenne_2_PhantomMesh_025.stl',
            'Antenne_3_PhantomMesh_026.stl',
            'Baffle_001_Cube_001.stl',
            'Baffle_002_Cube_003.stl',
            'Baffle_003_Cube_005.stl',
            'Baffle_004_Cube_029.stl',
            'Baffle_005_Cube_006.stl',
            'Baffle_006_Cube_007.stl',
            'Baffle_007_Cube_008.stl',
            'Baffle_008_Cube_011.stl',
            'Baffle_010_Cube_013.stl',
            'Baffle_013_Cube_043.stl',
            'Baffle_014_Cube_044.stl',
            'Baffle_015_Cube_045.stl',
            'Boitier_Div_001_W31-1-1-SUPPORT-BOBINE-SS-ELEC_006_W31-1-1-SUPP.stl',
            'Boitier_Div_002_W31-1-1-SUPPORT-BOBINE-SS-ELEC_006_W31-1-1-SUPP.stl',
            'Boitier_Div_003_W31-1-1-SUPPORT-BOBINE-SS-ELEC_006_W31-1-1-SUPP.stl',
            'Boitier_Div_004_W31-1-1-SUPPORT-BOBINE-SS-ELEC_006_W31-1-1-SUPP.stl',
            'Boitier_Div_005_W31-1-1-SUPPORT-BOBINE-SS-ELEC_006_W31-1-1-SUPP.stl',
            'Boitier_Div_007_W31-1-1-SUPPORT-BOBINE-SS-ELEC_006_W31-1-1-SUPP.stl',
            'Chemin_e_001_BRep_198_LOD0_low_BRep_198_LOD0_028.stl',
            'Chemin_e_002_BRep_198_LOD0_low_BRep_198_LOD0_029.stl',
            'Chemin_e_003_BRep_198_LOD0_low_BRep_198_LOD0_030.stl',
            'Chemin_e_004_BRep_198_LOD0_low_BRep_198_LOD0_031.stl',
            'Chemin_e_005_BRep_198_LOD0_low_BRep_198_LOD0_032.stl',
            'Chemin_e_006_BRep_198_LOD0_low_BRep_198_LOD0_033.stl',
            'Chemin_e_007_BRep_198_LOD0_low_BRep_198_LOD0_007.stl',
            'Chemin_e_008_BRep_198_LOD0_low_BRep_198_LOD0_034.stl',
            'Chemin_e_009_BRep_198_LOD0_low_BRep_198_LOD0_035.stl',
            'Chemin_e_010_BRep_198_LOD0_low_BRep_198_LOD0_036.stl',
            'Chemin_e_011_BRep_198_LOD0_low_BRep_198_LOD0_037.stl',
            'Chemin_e_012_BRep_198_LOD0_low_BRep_198_LOD0_038.stl',
            'Chemin_e_013_BRep_198_LOD0_low_BRep_198_LOD0_039.stl',
            'Chemin_e_014_BRep_198_LOD0_low_BRep_198_LOD0_040.stl',
            'Chemin_e_015_BRep_198_LOD0_low_BRep_198_LOD0_041.stl',
            'Chemin_e_016_BRep_198_LOD0_low_BRep_198_LOD0_042.stl',
            'Chemin_e_017_BRep_198_LOD0_low_BRep_198_LOD0_043.stl',
            'Chemin_e_018_BRep_198_LOD0_low_BRep_198_LOD0_044.stl',
            'Chemin_e_019_BRep_198_LOD0_low_BRep_198_LOD0_045.stl',
            'Chemin_e_020_BRep_198_LOD0_low_BRep_198_LOD0_046.stl',
            'Chemin_e_021_BRep_198_LOD0_low_BRep_198_LOD0_047.stl',
            'Chemin_e_025_BRep_198_LOD0_low_BRep_198_LOD0_025.stl',
            'Chemin_e_026_BRep_198_LOD0_low_BRep_198_LOD0_026.stl',
            'Chemin_e_027_BRep_198_LOD0_low_BRep_198_LOD0_027.stl',
            'Enceinte___vide_1_001_VVC_5002_09-J5_A_ASM__B_2_001_Cylinder_0.stl',
            'Enceinte___vide_1_007_VVC_5002_09-J5_A_ASM__B_2_001_Cylinder_0.stl',
            'Enceinte___vide_1_008_VVC_5002_09-J5_A_ASM__B_2_001_Cylinder_0.stl',
            'Enceinte___vide_1_009_VVC_5002_09-J5_A_ASM__B_2_001_Cylinder_0.stl',
            'Enceinte___vide_1_010_VVC_5002_09-J5_A_ASM__B_2_001_Cylinder_0.stl',
            'Enceinte___vide_1_011_VVC_5002_09-J5_A_ASM__B_2_001_Cylinder_0.stl',
            'Enceinte___vide_2_001_VVC_5002_09-J5_A_ASM__B_2_002_Cylinder_0.stl',
            'Enceinte___vide_2_002_VVC_5002_09-J5_A_ASM__B_2_002_Cylinder_0.stl',
            'Enceinte___vide_2_003_VVC_5002_09-J5_A_ASM__B_2_002_Cylinder_0.stl',
            'Enceinte___vide_2_004_VVC_5002_09-J5_A_ASM__B_2_002_Cylinder_0.stl',
            'Enceinte___vide_2_015_VVC_5002_09-J5_A_ASM__B_2_002_Cylinder_0.stl',
            'Enceinte___vide_2_016_VVC_5002_09-J5_A_ASM__B_2_002_Cylinder_0.stl',
            'Enceinte___vide_2_018_VVC_5002_09-J5_A_ASM__B_2_002_Cylinder_0.stl',
            'Enceinte___vide_2_019_VVC_5002_09-J5_A_ASM__B_2_002_Cylinder_0.stl',
            'Enceinte___vide_2_020_VVC_5002_09-J5_A_ASM__B_2_002_Cylinder_0.stl',
            'Enceinte___vide_2_022_VVC_5002_09-J5_A_ASM__B_2_002_Cylinder_0.stl',
            'Enceinte___vide_2_023_VVC_5002_09-J5_A_ASM__B_2_002_Cylinder_0.stl',
            'Enceinte___vide_2_024_VVC_5002_09-J5_A_ASM__B_2_002_Cylinder_0.stl',
            'PEI_Ext_3_001_BRep_1_002.stl',
            'PEI_Ext_3_002_BRep_1_004.stl',
            'PEI_Ext_3_003_BRep_1_005.stl',
            'PEI_Ext_3_004_BRep_1_006.stl',
            'PEI_Ext_3_005_BRep_1_007.stl',
            'PEI_Ext_3_006_BRep_1_008.stl',
            'PEI_Externe_1_001_IVP_3202_09-LFS_PROTECTION_Q3B_037.stl',
            'PEI_Externe_1_002_IVP_3202_09-LFS_PROTECTION_Q3B_038.stl',
            'PEI_Externe_1_003_IVP_3202_09-LFS_PROTECTION_Q3B_039.stl',
            'PEI_Externe_1_004_IVP_3202_09-LFS_PROTECTION_Q3B_040.stl',
            'PEI_Externe_1_005_IVP_3202_09-LFS_PROTECTION_Q3B_041.stl',
            'PEI_Externe_1_006_IVP_3202_09-LFS_PROTECTION_Q3B_042.stl',
            'PEI_Externe_1_007_IVP_3202_09-LFS_PROTECTION_Q3B_043.stl',
            'PEI_Externe_1_008_IVP_3202_09-LFS_PROTECTION_Q3B_033.stl',
            'PEI_Externe_1_009_IVP_3202_09-LFS_PROTECTION_Q3B_044.stl',
            'PEI_Externe_1_010_IVP_3202_09-LFS_PROTECTION_Q3B_045.stl',
            'PEI_Externe_1_011_IVP_3202_09-LFS_PROTECTION_Q3B_046.stl',
            'PEI_Externe_1_012_IVP_3202_09-LFS_PROTECTION_Q3B_047.stl',
            'PEIExterne_2_001_IVP_3202_07-LFS_PROTECTION_PJ3_002_IVP_3202_07.stl',
            'PEIExterne_2_002_IVP_3202_07-LFS_PROTECTION_PJ3_002_IVP_3202_07.stl',
            'PEIExterne_2_003_IVP_3202_07-LFS_PROTECTION_PJ3_002_IVP_3202_07.stl',
            'PEIExterne_2_004_IVP_3202_07-LFS_PROTECTION_PJ3_002_IVP_3202_07.stl',
            'PEIExterne_2_005_IVP_3202_07-LFS_PROTECTION_PJ3_002_IVP_3202_07.stl',
            'PEIExterne_2_IVP_3202_07-LFS_PROTECTION_PJ3_002_IVP_3202_07-LFS.stl',
            'PEI_interne_001_BRep_1_096_BRep_1_001.stl',
            'PEI_interne_002_BRep_1_096_BRep_1_002.stl',
            'PEI_interne_003_BRep_1_096_BRep_1_009.stl',
            'PEI_interne_004_BRep_1_096_BRep_1_003.stl',
            'PEI_interne_005_BRep_1_096_BRep_1_006.stl',
            'PEI_interne_006_BRep_1_096_BRep_1_013.stl',
            'PFU_Cu_001_Cube_385.stl',
            'PFU_Cu_002_Cube_087.stl',
            'PFU_Cu_003_Cube_386.stl',
            'PFU_Cu_004_Cube_089.stl',
            'PFU_Cu_005_Cube_090.stl',
            'PFU_Cu_006_Cube_091.stl',
            'PFU_Cu_007_Cube_387.stl',
            'PFU_Cu_008_Cube_093.stl',
            'PFU_Cu_009_Cube_388.stl',
            'PFU_Cu_010_Cube_095.stl',
            'PFU_Cu_011_Cube_389.stl',
            'PFU_Cu_012_Cube_097.stl',
            'PFU_Graphite_001_IVP_3205_04-INNER_LIMITERS_PJ4_037.stl',
            'PFU_Graphite_002_IVP_3205_04-INNER_LIMITERS_PJ4_038.stl',
            'PFU_Graphite_003_IVP_3205_04-INNER_LIMITERS_PJ4_009.stl',
            'PFU_Graphite_004_IVP_3205_04-INNER_LIMITERS_PJ4_039.stl',
            'PFU_Graphite_005_IVP_3205_04-INNER_LIMITERS_PJ4_040.stl',
            'PFU_Graphite_007_IVP_3205_04-INNER_LIMITERS_PJ4_041.stl',
            'PFU_W_001_Cube_018.stl',
            'PFU_W_002_Cube_020.stl',
            'PFU_W_003_Cube_023.stl',
            'PFU_W_004_Cube_025.stl',
            'PFU_W_005_Cube_027.stl',
            'PFU_W_006_Cube_030.stl',
            'PFU_W_008_Cube_032.stl',
            'PFU_W_009_Cube_033.stl',
            'PFU_W_010_Cube_034.stl',
            'PFU_W_011_Cube_014.stl',
            'PFU_W_014_Cube_036.stl',
            'PFU_W_027_Cube_016.stl',
            'Support_Baffle_001_Cube_009.stl',
            'Support_Baffle_002_Cube_004.stl',
            'Support_Baffle_003_Cube_010.stl',
            'Support_Baffle_004_Cube_012.stl',
            'Support_Baffle_005_Cube_028.stl',
            'Support_Baffle_007_Cube_046.stl']
roughness = 0.1
stl_base_dir = '/home/skosslow/west-stl'
for i in range(0,len(stl_list)):
    q = import_stl(stl_base_dir + '/' + stl_list[i],scaling=1.0,mode='auto',parent=world)
    q.material = RoughTungsten(roughness)#Lambert(wSF)#AbsorbingSurface()#
    westmesh.append(q)

## Create Mesh and Plasma from edge2d objects
cells_vol=0
cells_area=0
dat = nc.Dataset('w-56854-ext-W-1m-divertor.nc')
# # Turn on for debugging to count the number of valid cells
# for ir in range(dat['NRS'][:]):
#     for ik in range(dat['NKS'][ir]):
#         if dat['KVOLS'][ir,ik] != 0.0:
#             cells_vol = cells_vol+1
#         if dat['KAREAS'][ir,ik] != 0.0:
#             cells_area = cells_area+1

# print('Volume Counted Cells:')
# print(cells_vol)
# print('Area Counted Cells:')
# print(cells_area)

ring_index = dat['NRS'][:]
knot_index = dat['NKS'][:]
npart = dat['ABSFAC'][:]

r_corners = np.empty([4,])
z_corners = np.empty_like(r_corners)

# flat_count=0
flat_vol = dat['KVOLS'][:,:].flatten()
# for iv in range(len(flat_vol)):
#     if flat_vol[iv] != 0.0:
#         flat_count = flat_count+1

# print('Flattened Volume Counted Cells:')
# print(flat_count)

for ir in range(ring_index):
    for ik in range(knot_index[ir]):
        index = dat['KORPG'][ir,ik]-1
        if dat['KVOLS'][ir,ik] != 0.0: # np.sum(dat['RVERTP'][iv,:]) != 0 and np.sum((dat['ZVERTP'][iv,:])) != 0:
            r_corners = np.vstack((r_corners, dat['RVERTP'][index][0:4]))
            z_corners = np.vstack((z_corners, dat['ZVERTP'][index][0:4]))

r_corners = np.delete(r_corners,0,axis=0)
z_corners = np.delete(z_corners,0,axis=0)
r_corners = r_corners.T
z_corners = z_corners.T

vol = np.empty([])
# for ir in range(dat['MAXNRS'][:]):
#     for ik in range(dat['MAXNKS'][:]):
# for ir in range(dat['NRS'][:]):
#     for ik in range(dat['NKS'][ir]):
for iv in range(len(dat['RVERTP'][:,:])):    
    if flat_vol[iv] != 0.0:
        vol = np.append(vol,flat_vol[iv]) # Cell volumes

vol = np.delete(vol,0)

mesh = Edge2DMesh(r_corners,z_corners,vol)

# Load the plasma species in the simulation
# Be sure to include the impurity in your plasma here
_popular_species = {
    (1,2): deuterium,
    (74,183.84): tungsten, #This may cause an error if tungsten is initialized with the wrong mass
}

species_list = []
species_list.append((deuterium,0))
species_list.append((deuterium,1))
nizs = dat['NIZS'][:]

neutral_list = []
neutral_list.append((deuterium,0))

imp_atomic_num = int(dat['CION'][:])
imp_mass = float(dat['CRMI'][:])
imp_mass = round(imp_mass, 2)

## OEDGE only tracks one impurity at a time, and additonal loop would need to be added when woring with multiple impurities
for i in range(nizs+1): # Add impurity charge states to species list
    zn = imp_atomic_num
    am = imp_mass
    #species_key = 
    charge = i
    species = _popular_species[(zn,am)]
    species_list.append((species.name, charge))
    if i == 0:
        neutral_list.append((species.name ,charge))

sim = Edge2DSimulation(mesh, species_list)

te = np.zeros(mesh.n)
ne = np.zeros(mesh.n)
ti = np.zeros(mesh.n)
d1_dens = np.zeros(mesh.n)
d0_dens = np.zeros(mesh.n)
d0_temp = np.zeros(mesh.n)
halpha = np.zeros(mesh.n)
imp0_temp = np.zeros(mesh.n)

## Debugging values
# count = 0
# for ir in range(ring_index):
#     for ik in range(knot_index[ir]):
#         if dat['KVOLS'][ir,ik] != 0.0:
#             if ik == 0:
#                 te[count] = 1e24
#             count = count + 1

# for ir in range(dat['MAXNRS'][:]):
#     for ik in range(dat['MAXNKS'][:]):
count = 0
for ir in range(ring_index):
    for ik in range(knot_index[ir]):
        if dat['KVOLS'][ir,ik] != 0.0:
            te[count] = dat['KTEBS'][ir,ik] #electron temp
            ne[count] = dat['KNBS'][ir,ik] #electron density
            ti[count] = dat['KTIBS'][ir,ik] #ion temp
            d0_dens[count] = dat['PINATO'][ir,ik]
            d0_temp[count] = dat['PINENA'][ir,ik]
            halpha[count] = dat['PINALP'][ir,ik] #D-alpha emission
            imp0_temp[count] = dat['DDTS'][1,ir,ik] 
            count = count + 1    

            # te = np.append(te,dat['KTEBS'][ir,ik]) #electron temp
            # ne = np.append(ne,dat['KNBS'][ir,ik]) #electron density
            # ti = np.append(ti,dat['KTIBS'][ir,ik]) #ion temp
            # d0_dens = np.append(d0_dens,dat['PINATO'][ir,ik])
            # d0_temp = np.append(d0_temp,dat['PINENA'][ir,ik])
            # halpha = np.append(halpha,dat['PINALP'][ir,ik]) #D-alpha emission
            # w0_temp = np.append(w0_temp,dat['DDTS'][0,ir,ik])
            # count = count + 1
d1_dens = ne #ion denstiy; OEDGE assumes that ni=ne in a hydrogenic plasma

neutral_temp_array = np.vstack((d0_temp,imp0_temp))

## Populate the species density array
species_density = np.zeros((len(species_list),mesh.n))
species_density[0,:] = d0_dens.flatten() # .flatten() may be redundant here
species_density[1,:] = d1_dens
# This nonsense is from stackoverflow and doesn't really work
try:
    imp_dens = dat['DDLIMS'][:,:,:]
except NameError:
    impurity_data_exists = False
else:
    impurity_data_exists = True


if impurity_data_exists == True:
    for ns in range(2,len(species_list)):
        count = 0
        # charge_imp_dens = np.empty([])
        for ir in range(ring_index):
            for ik in range(knot_index[ir]):
                if dat['KVOLS'][ir,ik] != 0.0:
                    # Populate a density array for each species, then stack them into 1 array with the D0 and D1 densities
                    # charge_imp_dens[count] = imp_dens[ns,ir,ik] # array for the 
                    species_density[ns,count] = imp_dens[ns-1,ir,ik]*npart
                    count = count + 1
        # species_density = np.vstack((species_density, charge_imp_dens)) # add the distrubution for each species to the array

b_field_ratio_in_cell = np.zeros(mesh.n)
plasma_v_parallel = np.zeros(mesh.n)
imp_v_parallel = np.zeros((nizs+1,mesh.n))
 
for ns in range(len(species_list)-2):
    count = 0
    for ir in range(ring_index):
        for ik in range(knot_index[ir]):
            if dat['KVOLS'][ir,ik] != 0.0:
                b_field_ratio_in_cell[count] = dat['BRATIO'][ir,ik]
                plasma_v_parallel[count] = dat['KVHS'][ir,ik]
                imp_v_parallel[ns,count] = dat['DDVS'][ns+1,ir,ik] # What do I do about this with the two neutral species?
                count = count + 1

theta = np.arcsin(b_field_ratio_in_cell) #Toroidal twisting angle to deconvolute velocity components in radians
plasma_v_pol = np.multiply(plasma_v_parallel,b_field_ratio_in_cell)#.flatten() #Poloidal velocity comp to deconvolute r z comps
plasma_v_tor = np.multiply(plasma_v_parallel,np.cos(theta)) #Toroidal velocity component
# v_tor = v_tor.flatten()
#phi angle calculation to deconvolute r z comps
plasma_v_r = np.zeros(mesh.n)
plasma_v_z = np.zeros(mesh.n)

imp_v_pol = np.zeros((nizs+1,mesh.n))
imp_v_tor = np.zeros((nizs+1,mesh.n))
for ns in range(nizs+1):
    imp_v_pol[ns,:] = np.multiply(imp_v_parallel[ns,:],b_field_ratio_in_cell)
    imp_v_tor[ns,:] = np.multiply(imp_v_parallel[ns,:],np.cos(theta))
imp_v_r = np.zeros((nizs+1,mesh.n))
imp_v_z = np.zeros((nizs+1,mesh.n))
phi = np.zeros(mesh.n)

count = 0
for ir in range(ring_index):
    for ik in range(knot_index[ir]):
        if dat['KVOLS'][ir,ik] != 0.0:
            backratio = (dat['ZS'][ir,ik]-dat['ZS'][ir,ik-1])/(dat['RS'][ir,ik]-dat['RS'][ir,ik-1])
            forwardratio = (dat['ZS'][ir,ik+1]-dat['ZS'][ir,ik])/(dat['RS'][ir,ik+1]-dat['RS'][ir,ik])
            phi[count] = 0.5*(np.arctan(backratio)+np.arctan(forwardratio))
            count = count + 1
            #This can be put into an array later if I want to be cute

plasma_v_r = plasma_v_pol*np.cos(phi)
plasma_v_z = plasma_v_pol*np.sin(phi)
for ns in range(nizs):
    imp_v_r[ns,:] = imp_v_pol[ns,:]*np.cos(phi)
    imp_v_z[ns,:] = imp_v_pol[ns,:]*np.sin(phi)

# Create one 2D array of the velocities
oedge_electron_velocity_profile = np.stack((plasma_v_r,plasma_v_tor,plasma_v_z)) 
oedge_species_velocity_profile = np.zeros((len(species_list),np.shape(oedge_electron_velocity_profile)[0],np.shape(oedge_electron_velocity_profile)[1]))
oedge_species_velocity_profile[0,:,:] = oedge_electron_velocity_profile
oedge_species_velocity_profile[1,:,:] = oedge_electron_velocity_profile # assumes ve = vi for ions and neutrals
for si in range(2,len(species_list)): #All of the species velocities are assumed to be the same so we stack the same array over and over -- This is not really true with the highly kinetic tungsten near the divertor
    oedge_species_velocity_profile[si,:,:] = np.stack((imp_v_r[ns,:],imp_v_tor[ns,:],imp_v_z[ns,:])) #,axis=1)

## These lines are obsolete, left for now for reference
#     # v_r.append(v_r_ring[:])
#     # v_r = np.append(v_r,v_r_ring)
#     # v_z.append(v_z_ring[:])
#     # v_z = np.append(v_z,v_z_ring)
# v_r = v_r.flatten()
# v_z = v_z.flatten()       

b_field = np.zeros((3,mesh.n))
b_tor = np.zeros(mesh.n) #toroidal magnetic field component
b_pol = np.zeros(mesh.n)

#b_r = np.empty([])
b_phi = np.zeros(mesh.n)
#b_z = np.empty([])

count = 0
for ir in range(ring_index):
    for ik in range(knot_index[ir]):
        if dat['KVOLS'][ir,ik] != 0.0:
            b_tor[count] = dat['BTS'][ir,ik]
            b_pol_to_b_tot = dat['BRATIO'][ir,ik]
            b_tot = b_tor[count] / np.sqrt(1. - b_pol_to_b_tot * b_pol_to_b_tot) 
            b_pol[count] = b_tot * b_pol_to_b_tot  # poloidal component

            ## We did this calculation before
            # backratio = (dat['ZS'][ir,ik]-dat['ZS'][ir,ik-1])/(dat['RS'][ir,ik]-dat['RS'][ir,ik-1]) 
            # forwardratio = (dat['ZS'][ir,ik+1]-dat['ZS'][ir,ik])/(dat['RS'][ir,ik+1]-dat['RS'][ir,ik]) 
            # phi = 0.5*(np.arctan(backratio)+np.arctan(forwardratio))

            #This can be put into an array later if I want to be cute
            b_phi[count] = dat['BTS'][ir,ik]

# b_tor = np.delete(b_tor,0)
# b_pol = np.delete(b_pol,0)
             
b_r = b_pol*np.cos(phi)
b_z = b_pol*np.sin(phi)

b_field[0,:] = b_r
b_field[1,:] = b_phi
b_field[2,:] = b_z      

#d0_temp_funny = np.reshape(d0_temp, (0,len(d0_temp)))

# Initialiaze the plasma simulation conditions
#sim.neutral_list = neutral_list
sim.electron_temperature = te
sim.electron_density = ne
sim.electron_velocities_cylindrical = oedge_electron_velocity_profile
sim.ion_temperature = ti
sim.neutral_temperature = neutral_temp_array 
sim.species_density = species_density
sim.velocities_cylindrical = oedge_species_velocity_profile
sim.b_field_cylindrical = b_field 
sim.halpha_radiation = halpha

## End of OEDGE integration, everything below is copied from w_plasma_v1.2_stable
#################################################################################

plasma = sim.create_plasma()
plasma.atomic_data = adas
plasma.parent = world

# (Redundant, automatically created) plasma.geometry = Cylinder(4.0,3.0,parent=world,transform=translate(0,0,-1.5))

# sim = SOLPSSimulation(mesh)
ni = len(mesh.r)
nj = len(mesh.z)

# Load electron species // commented out to see what an issue with mesh_data_dict comes from #Is this redundant with the new SOLPS read-in method?
#sim._electron_temperature = mesh_data_dict['te']/Q
# sim._electron_density = mesh_data_dict['ne']

# Load the plasma species in the simulation
## Unneeded old SOLPS method, kept for reference
""" _popular_species = {
    (1, 2): deuterium,
    (2, 4.003): helium,
    (2, 4.0): helium,
    (74, 184): tungsten, #This may cause an error if tungsten is initialized wrong
}

sim._species_list = []
for i in range(len(sim_info_dict['zn'])):

    zn = int(sim_info_dict['zn'][i])  # Nuclear charge
    am = float(sim_info_dict['am'][i])  # Atomic mass
    charge = int(sim_info_dict['zamax'][i])  # Ionisation/charge
    species = _popular_species[(zn, am)]
    sim.species_list.append(species.symbol + str(charge))

sim._species_density = mesh_data_dict['na'] """

## Create Plasma from SOLPS data distionary

## Initialize plasma species densities and temperatures
print('Begin plasma setup')
# me = mesh.mesh_extent ## Unnecessary 8/25/2022 

## This section might be unneccessary since I dont use the imshow plotting method any more
xl, xu = (np.min(mesh.r), np.max(mesh.r))
yl, yu = (np.min(mesh.z), np.max(mesh.z))

D0 = plasma.composition.get(deuterium, 0)
D1 = plasma.composition.get(deuterium, 1)

W0 = plasma.composition.get(tungsten, 0)
W1 = plasma.composition.get(tungsten, 1)
W2 = plasma.composition.get(tungsten, 2)
W3 = plasma.composition.get(tungsten, 3)
W4 = plasma.composition.get(tungsten, 4)
W5 = plasma.composition.get(tungsten, 5)
W6 = plasma.composition.get(tungsten, 6)
W7 = plasma.composition.get(tungsten, 7)

## Define emission lines of D plasma
# D-I lines
d_alpha = Line(elements.deuterium, 0, (3,2))
d_beta = Line(elements.deuterium, 0, (4,2))
d_gamma = Line(elements.deuterium, 0, (5,2))
d_delta = Line(elements.deuterium, 0, (6,2))  # n = 6->2: 410.12nm
d_epsilon = Line(elements.deuterium, 0, (7,2))  # n = 7->2: 396.95nm

## Define emission lines of W plasma ions
# W-I lines
wi_4009 = Line(elements.tungsten, 0, ('59 - 5d46s6p(7p)','2 - 5p65d56s(7s)')) # You MUST go to the repository to get the correct transition indexes since the full spectroscopy configs weren't used
wi_4074a = Line(elements.tungsten, 0, ('173 - 5d36s26p(1p)','18 - 5p65d46s2(3d)'))
wi_4074b = Line(elements.tungsten, 0, ('54 - 5d46s6p(7d)','2 - 5p65d56s(7s)'))
wi_4295 = Line(elements.tungsten, 0, ('49 - 5d46s6p(7p)','2 - 5p65d56s(7s)'))

## Add emitters for each species
"""
THIS DOESN'T WORK BECAUSE OF plasma.inside_outside
d_ion_species = plasma.composition.get(deuterium, 1)
d_atom_species = plasma.composition.get(deuterium, 0)
wi_atom_species = plasma.composition.get(tungsten, 0)

d_alpha_excit = ExcitationLine(d_alpha, plasma.electron_distribution, d_atom_species, inside_outside=plasma.inside_outside)
d_alpha_excit.add_emitter_to_world(world, plasma)
d_alpha_recom = RecombinationLine(d_alpha, plasma.electron_distribution, d_ion_species, inside_outside=plasma.inside_outside)
d_alpha_recom.add_emitter_to_world(world, plasma)

d_gamma_excit = ExcitationLine(d_gamma, plasma.electron_distribution, d_atom_species, inside_outside=plasma.inside_outside)
d_gamma_excit.add_emitter_to_world(world, plasma)
d_gamma_recom = RecombinationLine(d_gamma, plasma.electron_distribution, d_ion_species, inside_outside=plasma.inside_outside)
d_gamma_recom.add_emitter_to_world(world, plasma)

d_beta_excit = ExcitationLine(d_beta, plasma.electron_distribution, d_atom_species, inside_outside=plasma.inside_outside)
d_beta_excit.add_emitter_to_world(world, plasma)
d_beta_recom = RecombinationLine(d_beta, plasma.electron_distribution, d_ion_species, inside_outside=plasma.inside_outside)
d_beta_recom.add_emitter_to_world(world, plasma)

d_delta_excit = ExcitationLine(d_delta, plasma.electron_distribution, d_atom_species, inside_outside=plasma.inside_outside)
d_delta_excit.add_emitter_to_world(world, plasma)
d_delta_recom = RecombinationLine(d_delta, plasma.electron_distribution, d_ion_species, inside_outside=plasma.inside_outside)
d_delta_recom.add_emitter_to_world(world, plasma)

d_epsilon_excit = ExcitationLine(d_epsilon, plasma.electron_distribution, d_atom_species, inside_outside=plasma.inside_outside)
d_epsilon_excit.add_emitter_to_world(world, plasma)
d_epsilon_recom = RecombinationLine(d_epsilon, plasma.electron_distribution, d_ion_species, inside_outside=plasma.inside_outside)
d_epsilon_recom.add_emitter_to_world(world, plasma)

wi_4009_excit = ExcitationLine(wi_4009, plasma.electron_distribution, wi_atom_species, inside_outside=plasma.inside_outside)
wi_4009_excit.add_emitter_to_world(world, plasma)

wi_4074a_excit = ExcitationLine(wi_4074a, plasma.electron_distribution, wi_atom_species, inside_outside=plasma.inside_outside)
wi_4074a_excit.add_emitter_to_world(world, plasma)

wi_4074b_excit = ExcitationLine(wi_4074b, plasma.electron_distribution, wi_atom_species, inside_outside=plasma.inside_outside)
wi_4074b_excit.add_emitter_to_world(world, plasma)

wi_4295_excit = ExcitationLine(wi_4295, plasma.electron_distribution, wi_atom_species, inside_outside=plasma.inside_outside)
wi_4295_excit.add_emitter_to_world(world, plasma)
 """

plasma.models = [
    Bremsstrahlung(),

    ExcitationLine(d_alpha, lineshape=StarkBroadenedLine),
    RecombinationLine(d_alpha, lineshape=StarkBroadenedLine),
    ExcitationLine(d_beta, lineshape=StarkBroadenedLine),
    RecombinationLine(d_beta, lineshape=StarkBroadenedLine),
    ExcitationLine(d_gamma, lineshape=StarkBroadenedLine),
    RecombinationLine(d_gamma, lineshape=StarkBroadenedLine),
    ExcitationLine(d_delta, lineshape=StarkBroadenedLine),
    RecombinationLine(d_delta, lineshape=StarkBroadenedLine),
    ExcitationLine(d_epsilon, lineshape=StarkBroadenedLine),
    RecombinationLine(d_epsilon, lineshape=StarkBroadenedLine),

    ExcitationLine(wi_4009), # lineshape=GaussianLine),
    ExcitationLine(wi_4074a), # lineshape=GaussianLine),
    ExcitationLine(wi_4074b), # lineshape=GaussianLine),
    ExcitationLine(wi_4295), # lineshape=GaussianLine),
]
print("End of plasma setup")

plt.ion()

print('Begin ray initialization and tracing')
## Create array of spectrometer cords and directions
# Try reading in all of the LOS from csv
wdw_rzp = np.zeros(3,)
wdw_rzp = np.array([2.9105,0.6876,2.687])
window = np.array([wdw_rzp[0]*np.cos(wdw_rzp[2]),wdw_rzp[0]*np.sin(wdw_rzp[2]),wdw_rzp[1]])
wdw_xyz = Point3D(window[0],window[1],window[2])
d2positions = pd.read_csv('~/projects/DVIS2coordinates.csv')
print(d2positions)
d2cords = d2positions.to_numpy()
rows = len(d2cords)

target = np.zeros((rows,3))
tgt_xyz = np.zeros((rows,3))

x = np.zeros(rows,)
y = np.zeros(rows,)
z = np.zeros(rows,)

#############################################
""" Set up cameras and spectroscope observer pipelines """
rgb = RGBPipeline2D()
spectral_power = SpectralPowerPipeline0D()
spectral_radiance = SpectralRadiancePipeline0D(accumulate=False) #Can switch observer accumulation on or off here
power = PowerPipeline2D()
radiance = RadiancePipeline0D()

print('Observing DVIS2 cords')
print('Create Group Observer')

# DVIS2 = FibreOpticGroup(parent=world)
# # Initializing fibre optic cords with 3D locations 
# for j in range (len(d2cords)): #Need to look at coordinates data file to set the loop range here (full range = 18 for shot 56854 dvis2)
#     x = d2cords[j,2]*np.cos(d2cords[j,4])
#     y = d2cords[j,2]*np.sin(d2cords[j,4])
#     z = d2cords[j,3]
#     tgt_xyz[j,0] = x
#     tgt_xyz[j,1] = y
#     tgt_xyz[j,2] = z
#     target = Point3D(x,y,z)
#     DVIS2.add_sight_line(SpectroscopicFibreOptic(wdw_xyz,wdw_xyz.vector_to(target),name=str(d2cords[j,1])))
# 
# DVIS2.acceptance_angle = 0.184
# DVIS2.radius = 2.e-3    
# DVIS2.min_wavelength = 380
# DVIS2.max_wavelength = 450
# DVIS2.spectral_bins = 3500
# DVIS2.pixel_samples = 5000
# DVIS2.spectral_rays = 1
# # DVIS2.render_engine = SerialEngine() # Run with one cpu core 
# DVIS2.render_engine = MulticoreEngine() # Run with all available cpu cores
# 
# DVIS2.observe()

## Curt's 2D emissivity plotting thing
spectrometer2D = FibreOpticGroup(parent=world)
start_point = Point3D(wdw_rzp[0],0,wdw_rzp[1])
up_vector = Vector3D(0,0,1.0)
for j in range(len(d2cords)):
    target = Point3D(d2cords[j,2],0,d2cords[j,3])
    forward_vector = start_point.vector_to(target)
    spectrometer2D.add_sight_line(SpectroscopicFibreOptic(start_point, forward_vector)) #start_point,acceptance_angle=1, radius=0.001, spectral_bins=8000, spectral_rays=1,
                   # pixel_samples=5, transform=translate(*start_point)*rotate_basis(forward_vector, up_vector)))

# spectrometer2D.transform = translate(*start_point)*rotate_basis(forward_vector, up_vector)
spectrometer2D.acceptance_angle = 0.184
spectrometer2D.radius = 0.001
spectrometer2D.spectral_bins = 8000
spectrometer2D.spectral_rays = 1
spectrometer2D.pixel_samples = 5
spectrometer2D.min_wavelength = 400.75
spectrometer2D.max_wavelength = 401.0
spectrometer2D.render_engine= MulticoreEngine()

# spectrometer2D.observe()
# spectrometer2D.plot_total_signal()

t_samples = np.arange(0, 1, 0.001)
# Setup some containers for useful parameters along the ray trajectory
ray_r_points = np.zeros((len(t_samples)))
ray_z_points = np.zeros((len(t_samples)))
distance = np.zeros((len(t_samples)))
te = np.zeros((len(t_samples)))
ne = np.zeros((len(t_samples)))
n_wi = np.zeros((len(t_samples)))
w4009 = np.zeros((len(t_samples),len(d2cords)))

# get the electron distribution
electrons = plasma.electron_distribution
wi = W0.distribution

# Pulling the PEC array to do interp2d on it. I need this to calculate the emissivity at each point
wi4009excit = adas.impact_excitation_pec(tungsten, 0, ('59 - 5d46s6p(7p)','2 - 5p65d56s(7s)'))

## Ray parameterization stuff taken from demo: https://www.cherab.info/demonstrations/line_emission/balmer_series_spectra.html#balmer-series-spectra
# Find the next intersection point of the ray with the world
for j in range(len(d2cords)):
    target = Point3D(d2cords[j,2],0,d2cords[j,3])
    forward_vector = start_point.vector_to(target)
    intersection = world.hit(Ray(start_point, forward_vector))
    if intersection is not None:
        hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
    else:
        raise RuntimeError("No intersection with the vessel was found.")
    ## An error raised here likely means the walls have been turned off
    # Traverse the ray with equation for a parametric line,
    # i.e. t=0->1 traverses the ray path.
    parametric_vector = start_point.vector_to(hit_point)
    # At each ray position sample the parameters of interest.
    for i, t in enumerate(t_samples):
        # Get new sample point location and log distance
        x = start_point.x + parametric_vector.x * t
        y = start_point.y + parametric_vector.y * t
        z = start_point.z + parametric_vector.z * t
        sample_point = Point3D(x, y, z)
        ray_r_points[i] = (np.sqrt(x**2 + y**2))
        ray_z_points[i] = (z)
        distance[i]=(start_point.distance_to(sample_point))

        # Sample plasma conditions
        te[i]=electrons.effective_temperature(x, y, z)
        ne[i]=(electrons.density(x, y, z))
        n_wi[i]=(wi.density(x, y, z))
        w4009[i,j] = ne[i] * n_wi[i] * wi4009excit(ne[i], te[i])
    plt.figure()
    plt.plot(distance,w4009[:,j],label=d2cords[j,1])
    plt.xlabel("Ray distance (m)")
    plt.ylabel("Normalised emission")
    plt.title("Normalised emission along ray path")
    plt.legend()

# Normalise the emission arrays

# dalpha /= dalpha.sum()
# dgamma /= dgamma.sum()
# dbeta /= dbeta.sum()
# ddelta /= ddelta.sum()
# depsilon /= depsilon.sum()
# w4009 /= w4009.sum()
# w4074 /= w4074.sum()
# w4295 /= w4295.sum()

# Plot the trajectory parameters 
# sim.plot_pec_emission_lines([w4009], title='Total Emission')
# plt.plot(R,Z,'k-',linewidth=1)
# plt.plot(ray_r_points, ray_z_points, 'g')
# plt.plot(ray_r_points[0], ray_z_points[0], 'b.')
# plt.plot(ray_r_points[-1], ray_z_points[-1], 'r.')

# plt.figure()
# plt.plot(distance, te)
# plt.xlabel("Ray distance (m)")
# plt.ylabel("Electron temperature (eV)")
# plt.title("Electron temperature (eV) along ray path")

# plt.figure()
# plt.plot(distance, ne)
# plt.xlabel("Ray distance (m)")
# plt.ylabel("Electron density (m^-3)")
# plt.title("Electron density (m^-3) along ray path")

# plt.figure()
# plt.plot(distance, dalpha, label='Dalpha')
# plt.plot(distance, dgamma, label='Dgamma')
# plt.plot(distance, dbeta, label='Dbeta')
# plt.plot(distance, ddelta, label='Ddelta')
# plt.plot(distance, depsilon, label='Depsilon')
# plt.xlabel("Ray distance (m)")
# plt.ylabel("Normalised emission")
# plt.title("Normalised emission along ray path")
# plt.legend()

# plt.figure()
# plt.plot(distance, w4009, label='W-I 400.9nm')
# plt.plot(distance, w4074, label='W-I 407.4nm')
# plt.plot(distance, w4295, label='W-I 429.5nm')
# plt.xlabel("Ray distance (m)")
# plt.ylabel("Normalised emission")
# plt.title("Normalised emission along ray path")
# plt.legend()

#############################################
""" Make OEDGE data plots with EDGE2D plot_quadrangle_mesh function """

# ax1 = mesh.plot_quadrangle_mesh()
# ax1.set_title('Quadrangle OEDGE Mesh')
# # plt.plot(R,Z,'k-',linewidth=1)
# plt.xlim(1.8,3.0)
# plt.ylim(-0.75,0.75)
# # plt.show()

# ax2 = mesh.plot_quadrangle_mesh(edge2d_data=sim.electron_temperature)
# ax2.set_title('Electron Temperature')
# ne_norm = mpl.colors.LogNorm(vmin=min(sim.electron_temperature),vmax=max(sim.electron_temperature))
# plt.colorbar(mpl.cm.ScalarMappable(norm = ne_norm),location='right')
# plt.plot(R,Z,'k-',linewidth=1)
# # for j in range(rows):
# #     plt.plot((wdw_rzp[0],d2cords[j,2]), (wdw_rzp[1],d2cords[j,3]), 'b-', linewidth=1)
# plt.plot((3.25,1.8),(0.25,0.0),'g-',linewidth=1) #Adding approximate lines for upper and lower extent of the SIR diagnostic
# plt.plot((3.25,2.5),(-0.1,-0.86),'g-',linewidth=1)
# ax2.set_xlim(1.8,3.0)
# ax2.set_ylim(-0.75,0.75)
# plt.show()

# plt.figure(3)
# ax3 = mesh.plot_quadrangle_mesh(edge2d_data=sim.species_density[0,:])
# ax3.set_title('Neutral Density')
# plt.plot(R,Z,'k-',linewidth=1)
# plt.xlim(1.8,3.0)
# plt.ylim(-0.75,0.75)
# plt.show()

# plt.figure(4)
# ax3 = mesh.plot_quadrangle_mesh(edge2d_data=sim.species_density[1,:])
# ax3.set_title('Ion Density')
# plt.plot(R,Z,'k-',linewidth=1)
# plt.xlim(1.8,3.0)
# plt.ylim(-0.75,0.75)
# plt.show()

# plt.figure(5)
# ax3 = mesh.plot_quadrangle_mesh(edge2d_data=sim.species_density[2,:])
# ax3.set_title('W0+ Density')
# plt.plot(R,Z,'k-',linewidth=1)
# plt.xlim(1.8,3.0)
# plt.ylim(-0.75,0.75)
# plt.show()

# plt.figure(6)
# ax3 = mesh.plot_quadrangle_mesh(edge2d_data=sim.species_density[3,:])
# ax3.set_title('W1+ Density')
# plt.plot(R,Z,'k-',linewidth=1)
# plt.xlim(1.8,3.0)
# plt.ylim(-0.75,0.75)
# plt.show()

# plt.figure(7)
# ax3 = mesh.plot_quadrangle_mesh(edge2d_data=sim.species_density[4,:])
# ax3.set_title('W2+ Density')
# plt.plot(R,Z,'k-',linewidth=1)
# plt.xlim(1.8,3.0)
# plt.ylim(-0.75,0.75)
# plt.show()

# plt.figure(7)
# ax3 = mesh.plot_quadrangle_mesh(edge2d_data=sim.species_density[5,:])
# ax3.set_title('W3+ Density')
# plt.plot(R,Z,'k-',linewidth=1)
# plt.xlim(1.8,3.0)
# plt.ylim(-0.75,0.75)
# plt.show()

# plt.figure(8)
# ax3 = mesh.plot_quadrangle_mesh(edge2d_data=sim.species_density[6,:])
# ax3.set_title('W4+ Density')
# plt.plot(R,Z,'k-',linewidth=1)
# plt.xlim(1.8,3.0)
# plt.ylim(-0.75,0.75)
# plt.show()

# plt.figure(9)
# ax3 = mesh.plot_quadrangle_mesh(edge2d_data=sim.species_density[7,:])
# ax3.set_title('W5+ Density')
# plt.plot(R,Z,'k-',linewidth=1)
# plt.xlim(1.8,3.0)
# plt.ylim(-0.75,0.75)
# plt.show()

# plt.figure(10)
# ax3 = mesh.plot_quadrangle_mesh(edge2d_data=sim.species_density[8,:])
# ax3.set_title('W6+ Density')
# plt.plot(R,Z,'k-',linewidth=1)
# plt.xlim(1.8,3.0)
# plt.ylim(-0.75,0.75)
# plt.show()

# plt.figure(11)
# ax3 = mesh.plot_quadrangle_mesh(edge2d_data=sim.species_density[9,:])
# ax3.set_title('W7+ Density')
# plt.plot(R,Z,'k-',linewidth=1)
# plt.xlim(1.8,3.0)
# plt.ylim(-0.75,0.75)
# plt.show()

#############################################
""" Plotting synthetic spectra created with CHERAB alongside experimental data """
""" Plotting synthetic spectra created with CHERAB alongside experimental data """
# D4341_spatial_profile = np.zeros(18,)
# for i in range(18): #turn this shit into a loop
#     lambda_nm = np.asarray(DVIS2.pipelines[i][0].wavelengths)
#     dg_idx_1_low_array = min(np.where(lambda_nm>433.8)) 
#     dg_idx_1_high_array = max(np.where(lambda_nm<434.3))
#     dg_idx_1_low = min(dg_idx_1_low_array)
#     dg_idx_1_high = max(dg_idx_1_high_array)
#     intensity = np.asarray(DVIS2.pipelines[i][0].samples.mean)
#     cherab_output = np.column_stack((lambda_nm,intensity))
#     # hkl.dump(cherab_output,'LODIVOU7_12jul22.hkl',mode='w')
#     D4341_spatial_profile[i] = max(intensity[idx] for idx in range(dg_idx_1_low,dg_idx_1_high))
#hkl.dump(D4341_spatial_profile,'59350_D4341_LODIVOU', mode='w')
# plt.yscale('log')
# plt.title(d2cords[0,1])
# plt.close()   
# DVIS2.plot_spectra(in_photons=True)
# plt.yscale('log')
# #plt.savefig('D4341_spatial_12jul22.png')
# plt.show()
# # print('End ray tracing')

# Plotting single LOS normalized to D-gamma
# VISTA = pd.read_csv('LODIVOU17-56854-spectrum.csv')
# print(VISTA)
# VISTA = VISTA.to_numpy()
# dg_idx_2_low_array = min(np.where(VISTA[:,0]>433.8))
# dg_idx_2_high_array = max(np.where(VISTA[:,0]<434.3))
# dg_idx_2_low = min(dg_idx_2_low_array)
# dg_idx_2_high = max(dg_idx_2_high_array)
# VISTA_norm = VISTA[:,1]/max(VISTA[idx,1] for idx in range(dg_idx_2_low,dg_idx_2_high)) #Fix this to make a new array without overwriting VISTA, do the same for intensity?
# intensity = intensity/max(intensity[idx] for idx in range(dg_idx_1_low,dg_idx_1_high))

# plt.figure()
# plt.plot(lambda_nm,intensity,'b-',label='CHERAB')
# plt.plot(VISTA[:,0],VISTA_norm,'r-',label='Experiment')
# plt.xlim(380,450)
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Normalized Intensity (-)')
# plt.legend(['CHERAB','Experimental'])
# plt.title('D-Gamma Normalized Intensity over single Line of Sight')
# plt.show()

# plt.figure()
# plt.semilogy(lambda_nm,intensity,'b-',label='CHERAB')
# plt.semilogy(VISTA[:,0],VISTA_norm,'r-',label='Experiment')
# plt.xlim(380,450)
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Normalized Intensity (-)')
# plt.legend(['CHERAB','Experimental'])
# plt.title('D-Gamma Normalized Intensity over single Line of Sight')
# plt.show()

# ## Plot D-gamma spatial brightness at the divertor
# plt.figure()
# plt.plot(d2cords[:,2],D4341_spatial_profile,'b*-')
# plt.xlabel('R (m)')
# plt.ylabel('D-gamma absolute intensity')
# plt.title('D-gamma spatial distribution')
# plt.show()

# ## Plot a spatial brightness for w-4009 line at lower divertor

# w4009_spatial_profile = np.zeros(18,)
# for i in range(18): #turn this shit into a loop
#     lambda_nm = np.asarray(DVIS2.pipelines[i][0].wavelengths)
#     wi_idx_1_low_array = min(np.where(lambda_nm>400.5)) 
#     wi_idx_1_high_array = max(np.where(lambda_nm<401.3))
#     wi_idx_1_low = min(wi_idx_1_low_array)
#     wi_idx_1_high = max(wi_idx_1_high_array)
#     intensity = np.asarray(DVIS2.pipelines[i][0].samples.mean)
#     # cherab_output = np.column_stack((lambda_nm,intensity))
#     # hkl.dump(cherab_output,'LODIVOU7_12jul22.hkl',mode='w')
#     w4009_spatial_profile[i] = max(intensity[idx] for idx in range(wi_idx_1_low,wi_idx_1_high))

# w4009_experimental = pd.read_csv('w4009-56854-divprofile.csv')
# w4009_exp = w4009_experimental.to_numpy()

# plt.figure()
# plt.semilogy(d2cords[:,2],w4009_spatial_profile,'b*-')
# plt.xlabel('R (m)')
# plt.ylabel('W-I 400.9nm Brightness (W/str/m^2/nm)')
# plt.title('W-I 400.9nm Divertor Brightness Profile')
# plt.show()

# plt.figure()
# plt.plot(w4009_exp[:,0],w4009_exp[:,1],'ro-')
# plt.xlabel('R (m)')
# plt.ylabel('W-I 400.9nm Calibratd intensity (ph/str/m^2)')
# plt.title('W-I 400.9nm Divertor Brightness Profile')
# plt.show()
#############################################
""" Use a camera to look at the inside of the vessel
This takes 15h to render! don't use it unless you really need a picture of the inside of the vessel!! """
print("Observing inside of vessel with Pinhole Camera")
divcamera = PinholeCamera((512, 512), fov = 90, pipelines=[rgb,power], parent=world)
divcamera.transform = translate(-2.615,1.277,0.6876)*rotate_x(170)*rotate_y(20)*rotate_z(-60)
divcamera.pixel_samples = 150
divcamera.spectral_bins = 1
divcamera.render_engine = MulticoreEngine()
# divcamera.pipelines[:].display_progess = False
# divcamera.observe()
# rgb.save('divcamera.png') #Save camera output as .png

toc = time.perf_counter()
elapsedtime = toc-tic
print(f"Time elapsed:{elapsedtime} seconds")
