'''
to analyse arepo simulation
'''

# general import
import os
import sys
sys.path.append('/home/yy503/Desktop/num_util')
sys.path.append('/home/yy503/Desktop/simulation_code/arepoutil')
import time
        
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib.colors as colors
import healpy
import h5py
import astropy.units as u; import astropy.constants as c
from scipy import stats
import yt

import anal_sim_util 
import astro_util
import coord_util
import plot_util as pu
from const_util import *

import imp

from pylab import *

import utils as util
import snapHDF5 as ws


par_file = '/data/ERCblackholes3/yuxuan/BBH_ISM_proj/MCS_dwarf_ism/parameterfile_SN-usedvalues'

with open(par_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        var_name, value = line.split()

        if var_name=='UnitVelocity_in_cm_per_s':
            UnitVel = float(value)
        elif var_name=='UnitLength_in_cm':
            UnitLength = float(value)
        elif var_name=='UnitMass_in_g':
            UnitMass = float(value)
del lines, line, var_name, value

# code unit in cgs
UnitTime = UnitLength/UnitVel
UnitDensity = UnitMass/UnitLength**3.
UnitEnergy = UnitMass*UnitVel**2.
UnitPressure = UnitEnergy / UnitLength**3.

# constant
Grav = 6.6742e-8
CLIGHT = 3.e10
PROTONMASS = 1.6726e-24
ElECTRONMASS = 9.11e-28
BOLTZMANN = 1.38065e-16
THOMPSON = 6.65e-25
PI = np.pi
HYDROGEN_MASSFRAC = 0.76
GAMMA_MINUS1 = 5./3. - 1.

def arepo_read_part_data(snap):
    '''
    Read particle data 
    '''
    h = ws.snapshot_header(snap)
    a = h.time
    posstar =  ws.read_block(snap,"POS ", 4) * UnitLength * a
    massstar = ws.read_block(snap, "MASS", 4) * UnitMass
    agestar = ws.read_block(snap, "AGE2", 4)
    
    centerstar = np.mean( posstar*np.array( [massstar]).T , axis=0 )/np.mean(massstar, axis=0)
    posstar -= centerstar
    return centerstar, posstar, massstar, agestar
    

def arepo_projection_plot(snap, field, weight_field=None, orientation='z', dim = 8*kpc2cm):
    '''
    projection of AREPO simulation
    '''
    h = ws.snapshot_header(snap)
    a = h.time
    
    posg = ws.read_block(snap,"POS ", 0) * UnitLength * a
    massg = ws.read_block(snap,"MASS", 0) * UnitMass
    rhog = ws.read_block(snap,"RHO ", 0) * UnitDensity
    
    center, posstar, massstar, agestar = arepo_read_part_data(snap)
    
    volg = massg / rhog 
    hsml = 2.5 * (volg * 3 / 4 / np.pi)**(1/3) * a
    

    fe = ws.read_block(snap, "EMF ", 0)
    ne = rhog*fe/ElECTRONMASS
    #ne = ws.read_block(snap, "NE  ", 0)
    ug = ws.read_block(snap, "U   ", 0) * UnitEnergy / UnitMass
    temp = GAMMA_MINUS1 / BOLTZMANN * PROTONMASS * ug * 4.0 / (1. + 3. * HYDROGEN_MASSFRAC + 4. * HYDROGEN_MASSFRAC * ne)
    
    posg -= center

    x, y, z = posg.T
    if orientation=='z':
        posg = np.array([x, y, z]).T
    elif orientation=='y':
        posg = np.array([x, z, y]).T

    hDim = dim / 2
    extent=[-hDim, hDim, -hDim, hDim]
    incg = util.cutCubeIndices(posg, -hDim)
    
    if field=='density':
        f = massg[incg]
    elif field=='temperature':
        f = temp[incg] 
        print('temp', f)
        
    if weight_field=='mass':
        wf = massg[incg]
        
    if weight_field is None:
        proj = util.doProjection(posg[incg], f, hsml[incg], 256,256, dim, dim, dim)
    else:
        proj1 = util.doProjection(posg[incg], f*wf, hsml[incg], 256,256, dim, dim, dim)
        proj2 = util.doProjection(posg[incg], wf, hsml[incg], 256,256, dim, dim, dim)
        proj = proj1/proj2
        
    return extent, proj 
    
    
def SFH_inst(directory, op_idx_a):
    nop = len(op_idx_a)
    time_a = np.zeros( nop )
    SFR_a = np.zeros( nop )
    
    for iop, op_idx in enumerate( op_idx_a ):
        op_idx = anal_sim_util.op_idx_int2str(op_idx, 3)
        snap="%s/snap_%s"%(directory, op_idx)
        
        h = ws.snapshot_header(snap)
        a = h.time
        time_a[iop]=a
        
        sfr = ws.read_block(snap, "SFR ", 0)
        SFR_a[iop] = np.sum(sfr)

    return time_a, SFR_a
    
def SFH_from_txt(directory):
    # refer to Arepo doc / General usage / Diagnostic Output Files / sfr.txt for details
    sfrtxt = np.loadtxt(directory+'/sfr.txt' )
    time_a = sfrtxt[:,0]
    Ms_pss_a = sfrtxt[:,1]
    SFR_allc_a = sfrtxt[:,2]
    SFR_actc_a = sfrtxt[:,3]
    Ms_ass_a = sfrtxt[:,4]
    Ms_tot_a = sfrtxt[:,5]
    
    return time_a, SFR_allc_a, SFR_actc_a
    
def SFH_tavg(snap):
    _, posstar, massstar, agestar = arepo_read_part_data(snap)
    age_bin = np.linspace(0,1,100)
    SFR_a = pu.cal_sfr(agestar, massstar, age_bin) 
    SFR_a *= 1e10/1e9
    time_a = 1-age_bin[:-1]
    return time_a, SFR_a