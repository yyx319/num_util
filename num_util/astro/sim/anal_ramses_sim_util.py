'''
Util to analyse RAMSES simulation
'''
from cProfile import label
from lib2to3.pygram import python_grammar_no_print_statement
import re
import os
import sys
import warnings

import numpy as np 
import random
import matplotlib.pyplot as plt
import scipy 
from scipy.ndimage import gaussian_filter
import astropy.units as u; import astropy.constants as c
from scipy import stats


sys.path.append('/home/yy503/Desktop/simulation_code/ramses_tools/MakeMovies')
import movie_utils
from support_functions import *

import yt

sys.path.append('/home/yy503/Desktop/simulation_code/rascas-develop/py')
import domain as d
import mesh as m


import num_util
from num_util.const_util import *


sys.path.append('/home/yy503/Desktop/simulation_code/ramses_tools/simulation_projector/MyUtils')
import input_output_functions as iop



class ramses_gas:
    '''
    ramses gas objects
    '''
    def __init__(self, dens, v, p_th, refine, Z):
        self.dens = dens
        self.v = v 
        self.p_th = p_th 
        self.refine = refine 
        self.Z = Z 

#########################################
# get basic info from RAMSES simulation #
#########################################

def ramses_variable_info(sim_name):
    nvar = 7 # number of basic variable: density, vx, vy, vz, P_th, pass_sca1, pass_sca2
    if 'MHD' in sim_name:
        nvar += 6 # B_left (xyz), B_right (xyz)
    if 'RT' in sim_name:
        nvar += 3 # pass_sca3,4,5
    if 'CR' in sim_name:
        nvar += 1 # P_CR
    return nvar

def ramses_get_sim_info(sim_dir, sim_name, op_idx, halo_method, verbose=False, id=1, verbose_print=False):
    '''
    read the info of RAMSES simulation
    Input:
        - sim_dir: simulation suite directory
    '''
    # op_idx should be str
    if type(op_idx)==int or type(op_idx)==np.int64:
        op_idx = num_util.op_idx_int2str(op_idx, 5)
    elif type(op_idx)==str:
        op_idx = num_util.op_idx_checkstr(op_idx, 5)
    else:
        raise Exception('op_idx is not the required format')
    
    hydro_file_descriptor = '%s/%s/output_%s/hydro_file_descriptor.txt'%(sim_dir, sim_name, op_idx)
    
    with open(hydro_file_descriptor, 'r') as f:
        if verbose_print: print(f.read())
            
    info_file = '%s/%s/output_%s/info_%s.txt'%(sim_dir, sim_name, op_idx, op_idx)
    lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum = movie_utils.read_infofile(info_file)
    
    if halo_method is None:
        if verbose: print('No halo method, only return basic info')
        return lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum
    
    elif type( halo_method ) is dict:
        track_pos_halo = np.array( halo_method['halo_pos'] )
        track_vel_halo = np.array( halo_method['halo_vel'] )
        return lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum, track_pos_halo, track_vel_halo
    
    elif halo_method=='tracker_smartin_spiral':
        rec_readfile='%s/%s/output_%s/track_%s.txt'%(sim_dir, sim_name, op_idx, op_idx)
        with open(rec_readfile, 'r') as f:
            for i in range(14):
                f.readline()
            halo_vel = np.array( f.readline().split('=')[1].split() ).astype(float)
            f.readline()
            f.readline()
            halo_pos = np.array( f.readline().split('=')[1].split() ).astype(float) 
            rvir = float(f.readline().split('=')[1].split()[0])*lfac*cm2kpc
            halo_mass = float(f.readline().split('=')[1].split()[0])*dfac*lfac**3*g2Msun
            for i in range(9):
                f.readline()
            mstar = np.array( f.readline().split('=')[1].split() ).astype(float)*dfac*lfac**3*g2Msun
            
            halo_svar = {'rvir':rvir, 'halo_mass': halo_mass, 'mstar':mstar}
            
    
    elif halo_method=='tracker_smartin_spiral_v4.4':
        rec_readfile='%s/%s/output_%s/track_%s.txt'%(sim_dir, sim_name, op_idx, op_idx)
        trackfile_data = iop.read_trackfile(rec_readfile, with_part=True, separate_flows=True)

        Aza_a = trackfile_data['trackers']['Azahar-a']

        mvir = Aza_a['mvir_halo']['physical']
        halo_pos = Aza_a['xpos_track']
        halo_vel = Aza_a['vcom_track']

        halo_svar = {'Aza_a':Aza_a}

    elif halo_method=='tracker_smartin_spiral_merger_centre_v4.4':
        rec_readfile='%s/%s/output_%s/track_%s.txt'%(sim_dir, sim_name, op_idx, op_idx)
        trackfile_data = iop.read_trackfile(rec_readfile, with_part=True, separate_flows=True)

        Aza_a = trackfile_data['trackers']['Azahar-a']
        Aza_b = trackfile_data['trackers']['Azahar']
        Aza_c = trackfile_data['trackers']['Azahar-c']

        nactive=0
        mvir=0
        halo_pos=0
        halo_vel=0
           
        if Aza_a['active']:
            nactive+=1
            m1 = Aza_a['mvir_halo']['physical']
            mvir += m1*u.Msun
            halo_pos += Aza_a['xpos_track']
            halo_vel += Aza_a['vcom_track']*m1*u.Msun

        if Aza_b['active']:
            nactive+=1
            m2 = Aza_b['mvir_halo']['physical']
            mvir += m2*u.Msun
            halo_pos += Aza_b['xpos_track']
            halo_vel += Aza_b['vcom_track']*m2*u.Msun
            
        if Aza_c['active']:
            nactive+=1
            m3 = Aza_c['mvir_halo']['physical']
            mvir += m3*u.Msun
            halo_pos += Aza_c['xpos_track']
            halo_vel += Aza_c['vcom_track']*m3*u.Msun
            
        halo_pos /= nactive
        halo_vel /= mvir
        halo_vel = halo_vel.decompose().value

        if Aza_a['active']:
            r1 = np.sqrt( np.sum( ( Aza_a['xpos_track']-halo_pos)**2 ) )*lfac*cm2kpc
        else: 
            r1=np.NaN 
        if Aza_b['active']:
            r2 = np.sqrt( np.sum( ( Aza_b['xpos_track']-halo_pos)**2 ) )*lfac*cm2kpc
        else: 
            r2=np.NaN
        if Aza_c['active']:
            r3 = np.sqrt( np.sum( ( Aza_c['xpos_track']-halo_pos)**2 ) )*lfac*cm2kpc
        else:
            r3=np.NaN
            
        if Aza_a['active'] and Aza_b['active']:
            r12 = np.sqrt( np.sum( ( Aza_a['xpos_track']-Aza_b['xpos_track'] )**2 ) )*lfac*cm2kpc
        else:
            r12 = np.NaN       
        if Aza_a['active'] and Aza_c['active']:
            r13 = np.sqrt( np.sum( ( Aza_a['xpos_track']-Aza_c['xpos_track'] )**2 ) )*lfac*cm2kpc
        else:
            r13 = np.NaN     
        if Aza_b['active'] and Aza_c['active']:
            r23 = np.sqrt( np.sum( ( Aza_b['xpos_track']-Aza_c['xpos_track'] )**2 ) )*lfac*cm2kpc
        else:
            r23 = np.NaN 

        if (r12 is np.NaN) and (r13 is np.NaN) and (r23 is np.NaN):
            state='merged'
        elif np.nanmin([r12, r13, r23])<5:
            state='merging'
        elif np.nanmin([r12, r13, r23])>=5:
            state='separated'
        else:
            raise Exception('error')

        halo_svar= {'Aza_a':Aza_a, 'Aza_b': Aza_b, 'Aza_c':Aza_c, 'r1':r1, 'r2':r2, 'r3':r3, 'r12':r12, 'r13':r13, 'r23':r23, 'state':state }
        
    else: 
        raise Exception('anal_sim_util.get_sim_info: halo_method not found.')

    if verbose==False:
        return lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum, halo_pos, halo_vel
    else:
        return lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum, halo_pos, halo_vel, halo_svar


###############################
# read ramses simulation data #
###############################
def ramses_define_loading_fields_funct(select):
    '''
    This function defines the proper load fields to upload to memory
    Use this in combination with the yt.load function
    ds = yt.load(readf_route, fields=load_fields, extra_particle_fields=extra_particle_fields)  
    '''

    load_fields=[]
    if(select in ['HD+SfFb','HD+SfFbBoost','HD+SfNoFb','HD+Sf',
                  'HD+thSfFb','HD+thSfFbBoost','HD+thSfNoFb','HD+thSf']):
        load_fields=["Density",
                     "x-velocity","y-velocity","z-velocity",
                     "Pressure","refine","Metallicity"]
    elif (select in ['RT+SfFb', 'RT', 'RT+Boost'] ):
        load_fields=["Density",
                     "x-velocity","y-velocity","z-velocity",
                     "Pressure","refine","Metallicity",
                     "xHII","xHeII","xHeIII"
                     ]
    elif (select in ['MHD+SfFb','iMHD+SfFb','sMHD+SfFb', 'MHD','iMHD','sMHD']):
        load_fields=["Density",
                     "x-velocity","y-velocity","z-velocity",
                     "x-Bfield-left","y-Bfield-left","z-Bfield-left",
                     "x-Bfield-right","y-Bfield-right","z-Bfield-right",
                     "Pressure","refine","Metallicity"]
    elif (select in ['CRiMHD+SfFb', 'CRiMHD'] ):
        load_fields=["Density",
                     "x-velocity","y-velocity","z-velocity",
                     "x-Bfield-left","y-Bfield-left","z-Bfield-left",
                     "x-Bfield-right","y-Bfield-right","z-Bfield-right",
                     "CR_Pressure","Pressure","refine","Metallicity"]
    elif (select in ['RTiMHD+SfFb','RTsMHD+SfFb', 'RTiMHD','RTsMHD', 'RTiMHD-new']):
        load_fields=["Density",
                     "x-velocity","y-velocity","z-velocity",
                     "x-Bfield-left","y-Bfield-left","z-Bfield-left",
                     "x-Bfield-right","y-Bfield-right","z-Bfield-right",
                     "Pressure","refine","Metallicity","xHII","xHeII","xHeIII"]
    elif (select in ['RTCRiMHD+SfFb','RTnsCRiMHD+SfFb', 'RTCRiMHD','RTnsCRiMHD']):
        load_fields=["Density",
                     "x-velocity","y-velocity","z-velocity",
                     "x-Bfield-left","y-Bfield-left","z-Bfield-left",
                     "x-Bfield-right","y-Bfield-right","z-Bfield-right",
                     "CR_Pressure","Pressure","refine","Metallicity",
                     "xHII","xHeII","xHeIII"]
    else:
        print( "Not found load type, loading default")
        load_fields=["Density",
                     "x-velocity","y-velocity","z-velocity",
                     "Pressure"]
    return load_fields


def ramses_define_var_name_funct(select):
    '''
    Define the name of variables used to store the field
    '''
    var_names=[]
    if(select in ['HD+SfFb','HD+SfFbBoost','HD+SfNoFb','HD+Sf','HD+thSfFb','HD+thSfFbBoost','HD+thSfNoFb','HD+thSf']):
        var_names=["dens",
                     "vx","vy","vz",
                     "p_th","refine","Z"]
    elif (select in ['RT+SfFb', 'RT', 'RT+Boost'] ):
        var_names=["dens",
                     "vx","vy","vz",
                     "p_th","refine","Z",
                     "xHII","xHeII","xHeIII"
                     ]
    elif (select in ['MHD+SfFb','iMHD+SfFb','sMHD+SfFb', 'MHD','iMHD','sMHD']):
        var_names=["dens",
                     "vx","vy","vz",
                     "Bx_l","By_l","Bz_l",
                     "Bx_r","By_r","Bz_r",
                     "p_th","refine","Z"]
    elif (select in ['CRiMHD+SfFb', 'CRiMHD'] ):
        var_names=["dens",
                     "vx","vy","vz",
                     "Bx_l","By_l","Bz_l",
                     "Bx_r","By_r","Bz_r",
                     "p_cr","p_th","refine","Z"]
    elif (select in ['RTiMHD+SfFb','RTsMHD+SfFb', 'RTiMHD','RTsMHD']):
        var_names=["dens",
                     "vx","vy","vz",
                     "Bx_l","By_l","Bz_l",
                     "Bx_r","By_r","Bz_r",
                     "p_th","refine","Z","xHII","xHeII","xHeIII"]
    elif (select in ['RTCRiMHD+SfFb','RTnsCRiMHD+SfFb', 'RTCRiMHD','RTnsCRiMHD']):
        var_names=["dens",
                     "vx","vy","vz",
                     "Bx_l","By_l","Bz_l",
                     "Bx_r","By_r","Bz_r",
                     "p_cr","p_th","refine","Z",
                     "xHII","xHeII","xHeIII"]
    else:
        print( "Not found load type, loading default")
        var_names=[ "dens",
                     "vx","vy","vz",
                     "p_th"]
    return var_names


def ramses_read_gas_data(dat_dir, sim_suite_dir, sim_name, op_idx, halo_method=None, method='yt', shape='sphere', verbose=False, **kwargs):
    '''
    halo_method:  
    method: method for extracting information cube, CDD, yt 
    '''
    # op_idx should be str
    if type(op_idx)==int or type(op_idx)==np.int64:
        op_idx = num_util.op_idx_int2str(op_idx, 5)
    elif type(op_idx)==str:
        op_idx = num_util.op_idx_checkstr(op_idx, 5)
    else:
        raise Exception('op_idx is not the required format')
    
    if halo_method is None:
        lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum = ramses_get_sim_info(sim_suite_dir, sim_name, op_idx, halo_method=halo_method)
        track_pos_halo = np.array([0.5, 0.5, 0.5])
        track_vel_halo = np.array([0, 0, 0])
    else:
        lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum, track_pos_halo, track_vel_halo = ramses_get_sim_info(sim_suite_dir, sim_name, op_idx, halo_method=halo_method)
    vmean_halo = track_vel_halo*lfac/tfac/1e5 # in km/s

    if method[:3]=='CDD':
        
        if 'cdd_dir' in kwargs.keys():
            cdd_dir = kwargs['cdd_dir']
        else:
            cdd_dir = '%s/%s/output%s/'%(dat_dir, sim_name, op_idx )
        
        cdom = d.domain.read( cdd_dir+"compute_domain.dom")
        mdom = m.mesh(filename= cdd_dir+"domain_1.mesh")
        # get T and nHI

        nleaf = mdom.nleaf
        
        xleaf = ( mdom.gas.xleaf[0,:] - cdom.center[0] )*lfac*cm2kpc
        yleaf = ( mdom.gas.xleaf[1,:] - cdom.center[1] )*lfac*cm2kpc
        zleaf = ( mdom.gas.xleaf[2,:] - cdom.center[2] )*lfac*cm2kpc

        lvleaf = mdom.gas.leaflevel
        volleaf = ( lfac / 2**lvleaf )**3 # pc^3
        Temp = mdom.gas.vth_sq_times_m*amu / (2.0*kb) # thermal T
        density    = mdom.gas.nhi
        massleaf = density*m_p*volleaf
        
        
        vxleaf = mdom.gas.vleaf[0,:]/1e5 # in km/s
        vyleaf = mdom.gas.vleaf[1,:]/1e5 
        vzleaf = mdom.gas.vleaf[2,:]/1e5 
        
        vleaf = np.stack( [vxleaf, vyleaf, vzleaf], axis=0 )
        vmean_gas = np.sum( vleaf*massleaf.reshape(1, nleaf), axis=1 )/np.sum( massleaf )
        
        
        if method.split('_')[1]=='gas':
            vmean = vmean_gas
        elif method.split('_')[1]=='halo':
            vmean = vmean_halo 
        
        vxleaf -= vmean[0] 
        vyleaf -= vmean[1]
        vzleaf -= vmean[2]
        
        rleaf = np.sqrt( xleaf**2 + yleaf**2 + zleaf**2 )
        vrleaf = (vxleaf*xleaf + vyleaf*yleaf + vzleaf*zleaf) / rleaf

        vleaf = np.sqrt( vxleaf**2 + vyleaf**2 + vzleaf**2 )
        ndust = mdom.gas.ndust
        return xleaf, yleaf, zleaf, vxleaf, vyleaf, vzleaf, rleaf, vleaf, vrleaf, lvleaf, volleaf, massleaf, Temp, density, ndust
    
    
    elif method=='yt':
        dvar = {}
        load_fields = ramses_define_loading_fields_funct(sim_name)
        var_names = ramses_define_var_name_funct(sim_name)

        if 'size_kpc' in kwargs.keys():
            size_kpc = kwargs['size_kpc']*u.kpc
        else:
            size_kpc = 20*u.kpc
        
        size=size_kpc/(lfac*cm2kpc*u.kpc)
        size=size.decompose().value
        if verbose: print('size in code unit', size)
        filename = '%s/%s/output_%s/info_%s.txt'%(sim_suite_dir, sim_name, op_idx, op_idx)
        extra_fields = [('tform', 'd'), ('metal', 'd'), ('imass', 'd') ]
        bbox = [track_pos_halo-size/2., track_pos_halo+size/2. ]
        ds = yt.load(filename, fields = load_fields, extra_particle_fields = extra_fields, bbox=bbox)
        ad = ds.all_data()

        # read amr coordinate and cell vol data
        x = ad['gas', 'x'].value/lfac - track_pos_halo[0] 
        y = ad['gas', 'y'].value/lfac - track_pos_halo[1]
        z = ad['gas', 'z'].value/lfac - track_pos_halo[2]
        x = x*lfac*cm2kpc *u.kpc 
        y = y*lfac*cm2kpc *u.kpc
        z = z*lfac*cm2kpc *u.kpc
        
        r = np.sqrt( x**2 + y**2 + z**2 )

        # select data inside of the box
        if shape=='sphere':    
            idx = np.where( r< size_kpc/2 )[0]
        elif shape=='cube':
            idx = np.where( (-size_kpc/2<x) & (x<size_kpc/2) & (-size_kpc/2<y) & (y<size_kpc/2) & (-size_kpc/2<z) & (z<size_kpc/2) )[0]
            
        dvar['r'] = r[idx]
        dvar['x'] = x[idx]
        dvar['y'] = y[idx]
        dvar['z'] = z[idx]
        
        cell_vol = ad['gas', 'cell_volume'].value *u.cm**3
        dvar['cell_vol'] = cell_vol[idx]

        # read amr field data
        for var_name, field in zip(var_names, load_fields):
            dvar[var_name] = ad['ramses', field][idx].value
        dvar['dens']=dvar['dens']*dfac *u.g/u.cm**3 
        
        dvar['vx'] = (dvar['vx']*lfac/tfac/1e5 - vmean_halo[0] )*u.km/u.s 
        dvar['vy'] = (dvar['vy']*lfac/tfac/1e5 - vmean_halo[1] )*u.km/u.s 
        dvar['vz'] = (dvar['vz']*lfac/tfac/1e5 - vmean_halo[2] )*u.km/u.s 
        dvar['vr'] = num_util.cal_vector_in_r( dvar['x'], dvar['y'], dvar['z'], dvar['vx'], dvar['vy'], dvar['vz'] )

        if 'Bx_l' in var_names:
            conversion_factor_B = np.sqrt(dfac) * lfac / tfac *aexp**2 *u.cm**(-1/2)*u.g**(1/2)*u.s**(-1) 
            dvar['Bx'] = (dvar['Bx_l']+dvar['Bx_r'])/2 * conversion_factor_B
            dvar['By'] = (dvar['By_l']+dvar['By_r'])/2 * conversion_factor_B
            dvar['Bz'] = (dvar['Bz_l']+dvar['Bz_r'])/2 * conversion_factor_B

            dvar['B'] = np.sqrt(dvar['Bx']**2 + dvar['By']**2 + dvar['Bz']**2)
        
        dvar['p_th'] = dvar['p_th']*dfac*lfac**2/tfac**2 *u.g/u.cm/u.s**2 
        if 'p_cr' in var_names:
            dvar['p_cr'] = dvar['p_cr']*dfac*lfac**2/tfac**2 *u.g/u.cm/u.s**2 
            
        
        return dvar


######################################################################################
def ramses_read_part_data(sim_suite_dir, sim_name, op_idx, size= 2e-3, halo_method='tracker_smartin_dwarf', unit='standard', domain_shape='box'):
    print('reading particle data')
    if type(op_idx)==int or type(op_idx)==np.int64:
        op_idx = num_util.op_idx_int2str(op_idx, 5)
    elif type(op_idx)==str:
        op_idx = num_util.op_idx_checkstr(op_idx, 5)
    else:
        raise Exception('op_idx is not the required format')

    if halo_method is None:
        lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum = ramses_get_sim_info(sim_suite_dir, sim_name, op_idx, halo_method=halo_method)
        track_pos_halo = np.array([0, 0, 0])
    else:
        lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum, track_pos_halo, track_vel_halo = ramses_get_sim_info(sim_suite_dir, sim_name, op_idx, halo_method=halo_method)

    load_fields = ramses_define_loading_fields_funct(sim_name)
    filename = '%s/%s/output_%s/info_%s.txt'%(sim_suite_dir, sim_name, op_idx, op_idx)
    extra_fields = [('tform', 'd'), ('metal', 'd'), ('imass', 'd') ]
    bbox = [track_pos_halo-size/2., track_pos_halo+size/2. ]
    ds = yt.load(filename, fields = load_fields, extra_particle_fields = extra_fields, bbox=bbox)
    current_time = (ds.current_time*ds.time_unit*s2Gyr).value
    Lbox = lfac*cm2kpc # box size or code length in kpc

    ad = ds.all_data()
    
    if unit=='original':
        p_imass = ad['all', 'imass'].value
        p_metal = ad['all', 'metal'] 
        pid = ad['all', 'particle_identity'] 
        p_mass = ad['all', 'particle_mass'].value # in g
        p_x = ad['all', 'particle_position_x'].value # in code unit
        p_y = ad['all', 'particle_position_y'].value 
        p_z = ad['all', 'particle_position_z'].value 
        p_rl = ad['all', 'particle_refinement_level']
        p_vx = ad['all', 'particle_velocity_x'] # cm/s
        p_vy = ad['all', 'particle_velocity_y']
        p_vz = ad['all', 'particle_velocity_z']
        p_tform = ad['all', 'tform']
    elif unit=='standard':
        p_imass = ad['all', 'imass'].value*dfac*lfac**3*g2Msun # in Msun
        p_metal = ad['all', 'metal'] 
        pid = ad['all', 'particle_identity'] 
        p_mass = ad['all', 'particle_mass'].value*g2Msun # in Msun 
        p_x = ( ad['all', 'particle_position_x'].value - track_pos_halo[0] ) * Lbox # in kpc
        p_y = ( ad['all', 'particle_position_y'].value - track_pos_halo[1] ) * Lbox
        p_z = ( ad['all', 'particle_position_z'].value - track_pos_halo[2] ) * Lbox
        p_rl = ad['all', 'particle_refinement_level']
        p_vx = ad['all', 'particle_velocity_x']/1e5 # km/s
        p_vy = ad['all', 'particle_velocity_y']/1e5
        p_vz = ad['all', 'particle_velocity_z']/1e5
        p_tform = ad['all', 'tform']
    else:
        raise Exception('no such unit system')
    size_kpc = size*Lbox
    
    # select particle in domain
    # for some reasons the yt select stars outside of the boundary
    if domain_shape=='box':    
        if unit=='original':
            idx = np.where( (p_x>track_pos_halo[0]-size/2) & (p_x<track_pos_halo[0]+size/2) & \
                            (p_y>track_pos_halo[1]-size/2) & (p_y<track_pos_halo[1]+size/2) & \
                            (p_z>track_pos_halo[2]-size/2) & (p_z<track_pos_halo[2]+size/2) )[0]
        elif unit=='standard':
            idx = np.where( (p_x>-size_kpc/2) & (p_x<size_kpc/2) & \
                            (p_y>-size_kpc/2) & (p_y<size_kpc/2) & \
                            (p_z>-size_kpc/2) & (p_z<size_kpc/2) )[0]

    elif domain_shape=='sphere':
        if unit=='original':
            p_r = np.sqrt( (p_x-track_pos_halo[0])**2 + (p_y-track_pos_halo[1])**2 + (p_z-track_pos_halo[2])**2 )
            idx = np.where( p_r < size/2 )[0]
        elif unit=='standard':
            p_r = np.sqrt( p_x**2 + p_y**2 + p_z**2 )
            idx = np.where( p_r < size_kpc/2 )[0]
    else:
        raise Exception('domain shape not included')
    
    p_imass = p_imass[idx]
    p_metal = p_metal[idx]
    pid = pid[idx]
    p_mass = p_mass[idx]
    p_x = p_x[idx]
    p_y = p_y[idx]
    p_z = p_z[idx]
    p_rl = p_rl[idx]
    p_vx = p_vx[idx]
    p_vy = p_vy[idx]
    p_vz = p_vz[idx]
    p_tform = p_tform[idx]

    # star particle 
    idx_star = np.where(p_metal!=0)[0] # 
    star_imass = p_imass[idx_star]
    star_metal = p_metal[idx_star] 
    star_id = pid[idx_star]
    star_mass = p_mass[idx_star]
    star_x = p_x[idx_star]
    star_y = p_y[idx_star]
    star_z = p_z[idx_star]
    star_rl = p_rl[idx_star]
    star_vx = p_vx[idx_star]
    star_vy = p_vy[idx_star]
    star_vz = p_vz[idx_star]
    star_age = p_tform[idx_star] 
    
    # star age is the time between the birth of star and current simulation time
    if 'RT' in sim_name:
        # star_age is the lookback time in tH, negative value
        star_age = np.array(star_age)*tH + cosmo.age(z=0).value # lookforward time
        star_age = ( current_time - star_age )*1000 # lookback time relative to the current simulation time in Myr, positive value, 
    else: 
        # TBW
        star_age = star_age

    # dm particle 
    idx_dm = np.where(p_metal==0)[0]
    dm_imass = p_imass[idx_dm]
    dm_metal = p_metal[idx_dm] 
    dm_id = pid[idx_dm]
    dm_mass = p_mass[idx_dm]
    dm_x = p_x[idx_dm]
    dm_y = p_y[idx_dm]
    dm_z = p_z[idx_dm]
    dm_rl = p_rl[idx_dm]
    dm_vx = p_vx[idx_dm]
    dm_vy = p_vy[idx_dm]
    dm_vz = p_vz[idx_dm]

    return p_imass, p_metal, pid, p_mass, p_x, p_y, p_z, p_rl, p_vx, p_vy, p_vz, p_tform, \
    star_imass, star_metal, star_id, star_mass, star_x, star_y, star_z, star_rl, star_vx, star_vy, star_vz, star_age, \
    dm_imass, dm_metal, dm_id, dm_mass, dm_x, dm_y, dm_z, dm_rl, dm_vx, dm_vy, dm_vz



