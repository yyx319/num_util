'''
to analyse arepo simulation
'''

# general import
import os
import sys
sys.path.append('/home/yy503/Desktop/simulation_code/arepoutil')
import time
        
import numpy as np 
import astropy.units as u; import astropy.constants as c
from scipy import stats
import yt

import num_util 
from num_util.const_util import *


from pylab import *

import utils as util
import snapHDF5 as ws

# constant in cgs
Grav = 6.6742e-8
CLIGHT = 3.e10
PROTONMASS = 1.6726e-24
ElECTRONMASS = 9.11e-28
BOLTZMANN = 1.38065e-16
THOMPSON = 6.65e-25
PI = np.pi
HYDROGEN_MASSFRAC = 0.76
GAMMA_MINUS1 = 5./3. - 1.


def arepo_read_unit(par_file):

    # read code unit in cgs
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

    UnitTime = UnitLength/UnitVel
    UnitDensity = UnitMass/UnitLength**3.
    UnitEnergy = UnitMass*UnitVel**2.
    UnitPressure = UnitEnergy / UnitLength**3.
    
    unit_dict = { 
        'UnitVel': UnitVel,
        'UnitLength': UnitLength,
        'UnitMass': UnitMass,
        'UnitTime': UnitTime,
        'UnitDensity': UnitDensity,
        'UnitEnergy': UnitEnergy,
        'UnitPressure': UnitPressure
    }
    
    print(unit_dict)
    return unit_dict 

def arepo_read_star_part_data(snap, unit_dict):
    '''
    Read star particle data 
    '''
    UnitVel, UnitLength, UnitMass, UnitTime, UnitDensity, UnitEnergy, UnitPressure = unit_dict.values()

    posstar =  ws.read_block(snap,"POS ", 4).astype('float64') * UnitLength 
    massstar = ws.read_block(snap, "MASS", 4).astype('float64') * UnitMass
    agestar = ws.read_block(snap, "AGE2", 4).astype('float64')
    
    centerstar = np.mean( posstar*np.array( [massstar]).T , axis=0 )/np.mean(massstar, axis=0)
    posstar -= centerstar
    return centerstar, posstar, massstar, agestar
    
def arepo_read_bh_part_data(snap, unit_dict):
    UnitVel, UnitLength, UnitMass, UnitTime, UnitDensity, UnitEnergy, UnitPressure = unit_dict.values()
    
    pos_bh = ws.read_block(snap,"POS ", 5).astype('float64') * UnitLength 
    mass_bh = ws.read_block(snap, "MASS", 5).astype('float64') * UnitMass
    
    print('BH', pos_bh, mass_bh)
    
    center_bh = np.sum( pos_bh*np.array( [mass_bh]).T , axis=0 )/np.sum(mass_bh, axis=0)
    pos_bh -= center_bh
    return center_bh, pos_bh, mass_bh 

def arepo_projection_plot(snap, unit_dict, field, weight_field=None, orientation='z', dim = 8*kpc2cm, select_center=None, verbose=False):
    '''
    projection of AREPO simulation, modified from Martin's code
    '''    
    UnitVel, UnitLength, UnitMass, UnitTime, UnitDensity, UnitEnergy, UnitPressure = unit_dict.values()
    
    posg = ws.read_block(snap,"POS ", 0).astype('float64') * UnitLength 
    massg = ws.read_block(snap,"MASS", 0).astype('float64') * UnitMass
    rhog = ws.read_block(snap,"RHO ", 0).astype('float64') * UnitDensity
    ug = ws.read_block(snap, "U   ", 0).astype('float64') * UnitEnergy / UnitMass
    try:
        fe = ws.read_block(snap, "EMF ", 0).astype('float64')
    except Exception as e:
        print(e)
        fe=np.zeros_like(massg)    
    
    if select_center is None:
        center=np.array([0,0,0])
    elif select_center=='star':
        center, posstar, massstar, agestar = arepo_read_star_part_data(snap, unit_dict)
    elif select_center=='gas':
        center=np.sum(posg*massg.reshape( len(massg), 1 ), axis=0) / np.sum(massg)
    elif select_center=='bh':
        center, pos_bh, mass_bh = arepo_read_bh_part_data(snap, unit_dict)
    if verbose: print(f'center={center}')
    
    posg -= center
    x, y, z = posg.T
    volg = massg / rhog 
    hsml = 2.5 * (volg * 3 / 4 / np.pi)**(1/3) 
    try:
        ne = ws.read_block(snap, "NE  ", 0).astype('float64')
    except:
        ne = rhog*fe/ElECTRONMASS
    temp = GAMMA_MINUS1 / BOLTZMANN * PROTONMASS * ug * 4.0 / (1. + 3. * HYDROGEN_MASSFRAC + 4. * HYDROGEN_MASSFRAC * ne)

    if orientation=='z':
        posg = np.array([x, y, z]).T
    elif orientation=='y':
        posg = np.array([x, z, y]).T

    hDim = dim / 2
    extent=np.array([-hDim, hDim, -hDim, hDim])
    incg = util.cutCubeIndices(posg, -hDim)
    
    if field=='density' and weight_field==None:
        f = massg[incg]
    elif field=='density' and weight_field=='density':
        f = rhog[incg]
        wf = massg[incg]
    elif field=='temperature' and weight_field=='density':
        f = temp[incg] 
        wf = massg[incg]
    else:
        raise Exception('No such combination (field, weight_field)')
        
        
    if verbose: print('do projection')
    if weight_field is None:
        proj = util.doProjection(posg[incg], f, hsml[incg], 256,256, dim, dim, dim)
    else:
        proj1 = util.doProjection(posg[incg], f*wf, hsml[incg], 256,256, dim, dim, dim)
        proj2 = util.doProjection(posg[incg], wf, hsml[incg], 256,256, dim, dim, dim)
        proj = proj1/proj2
        
    return extent, proj 
    
    
# SFH
def arepo_SFH_inst(directory, op_idx_a):
    nop = len(op_idx_a)
    time_a = np.zeros( nop )
    SFR_a = np.zeros( nop )
    
    for iop, op_idx in enumerate( op_idx_a ):
        op_idx = num_util.op_idx_int2str(op_idx, 3)
        snap="%s/snap_%s"%(directory, op_idx)
        
        h = ws.snapshot_header(snap)
        a = h.time
        time_a[iop]=a
        
        sfr = ws.read_block(snap, "SFR ", 0)
        SFR_a[iop] = np.sum(sfr)

    return time_a, SFR_a
    
def arepo_SFH_from_txt(directory):
    '''
    Refer to Arepo doc / General usage / Diagnostic Output Files / sfr.txt for details
    '''
    sfrtxt = np.loadtxt(directory+'/sfr.txt' )
    time_a = sfrtxt[:,0]
    Ms_pss_a = sfrtxt[:,1]
    SFR_allc_a = sfrtxt[:,2]
    SFR_actc_a = sfrtxt[:,3]
    Ms_ass_a = sfrtxt[:,4]
    Ms_tot_a = sfrtxt[:,5]
    
    return time_a, SFR_allc_a, SFR_actc_a
    
