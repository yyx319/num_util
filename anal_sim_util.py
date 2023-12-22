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

from scipy import signal
from scipy.io import FortranFile as ff
from astropy.cosmology import WMAP7
import yt

sys.path.append('/home/yy503/Desktop/simulation_code/rascas-develop/py')
import domain as d
import mesh as m

import astro_util
import coord_util
from const_util import *

h0 = 0.6790000152587891*100
tH = 1./(h0*u.km/u.Mpc/u.s)
tH = tH.to(u.Gyr).value # Hubble time in Gyr
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=h0, Om0=0.306499987840652, Tcmb0=2.725)


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

def variable_info(sim_name):
    nvar = 7 # number of basic variable: density, vx, vy, vz, P_th, pass_sca1, pass_sca2
    if 'MHD' in sim_name:
        nvar += 6 # B_left (xyz), B_right (xyz)
    if 'RT' in sim_name:
        nvar += 3 # pass_sca3,4,5
    if 'CR' in sim_name:
        nvar += 1 # P_CR
    return nvar

def op_idx_int2str(op_idx, L=5):
    '''
    covert op_idx from int to string
    Input: 
        op_idx: output index. int type
        L: total length of string
    Output:
        op_idx: output index. str type
    ex: op_idx=55 L=5 -> '00055'
    '''
    if type(op_idx)==int or type(op_idx)==np.int64:
        pass
    else:
        raise Exception('error, input op_idx is not int')
    op_idx = str(op_idx)
    if len(op_idx)<L:
        op_idx = '0'*(L-len(op_idx)) + op_idx
    elif len(op_idx)==L:
        pass
    elif len(op_idx)>5:
        raise Exception('error, output index is longer than the largest length')
    return op_idx


def op_idx_checkstr(op_idx, L=5):
    '''
    check if op_idx follow the str format, if not change it to required format
    '''
    if type(op_idx)!=str:
        raise Exception('op_idx is not a string')
    elif type(op_idx)==str:
        pass

    if len(op_idx)<L:
        op_idx = '0'*( L-len(op_idx) ) + op_idx
    elif len(op_idx)==L:
        pass
    elif len(op_idx)>L:
        raise Exception('error, output index is longer than the largest length')
    
    return op_idx



def get_sim_info(sim_dir, sim_name, op_idx, halo_method, verbose=False, id=1):
    '''
    read the info of RAMSES simulation
    '''
    # op_idx should be str
    if type(op_idx)==int or type(op_idx)==np.int64:
        op_idx = op_idx_int2str(op_idx, 5)
    elif type(op_idx)==str:
        op_idx = op_idx_checkstr(op_idx, 5)
    else:
        raise Exception('op_idx is not the required format')
    
    hydro_file_descriptor = '%s/%s/output_%s/hydro_file_descriptor.txt'%(sim_dir, sim_name, op_idx)
    with open(hydro_file_descriptor, 'r') as f:
        print(f.read())
            
    info_file = '%s/%s/output_%s/info_%s.txt'%(sim_dir, sim_name, op_idx, op_idx)
    lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum = movie_utils.read_infofile(info_file)
    
    if halo_method=='tracker_smartin_spiral':
            
        with open('%s/%s/output_%s/track_%s.txt'%(sim_dir, sim_name, op_idx, op_idx), "r") as f:
            # basic info
            ntrackers = int( f.readline().split('=')[1] )
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            
            for i in range(ntrackers):
                itrack = int( f.readline().split('=')[1] )
                f.readline()
                f.readline()
                f.readline()
                x_halo = float( f.readline()[14:24] )
                y_halo = float( f.readline()[14:24] )
                z_halo = float( f.readline()[14:24] )
                halo_vel = np.array( [float(vi) for vi in f.readline()[14:].split() ] )
                f.readline() 
                rvir = float( f.readline().split('=')[1] )
                mvir = float( f.readline().split('=')[1] )
                for j in range(24):
                    f.readline() 
                    
                halo_pos = np.array([x_halo, y_halo, z_halo])
                
                if itrack==id: 
                    print('get info of halo No. %d'%id )                
                    break

    elif halo_method=='tracker_smartin_dwarf':
        with open('%s/%s/output_%s/track_%s.txt'%(sim_dir, sim_name, op_idx, op_idx), "r") as f:
            # basic info
            ntrackers = int( f.readline().split('=')[1] )
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            itrack = int( f.readline().split('=')[1] )
            f.readline()
            f.readline()
            f.readline()
            x_halo = float( f.readline()[14:24] )
            y_halo = float( f.readline()[14:24] )
            z_halo = float( f.readline()[14:24] )
            halo_vel = np.array( [float(vi) for vi in f.readline()[14:].split() ] )
                
            halo_pos = np.array([x_halo, y_halo, z_halo])
            
            halo_svar = {'rvir':None, 'mvir':None}
            for i in range(100):
                try: 
                    var, value = f.readline().replace(' ', '').split('=')
                    if var in halo_svar.keys():
                        halo_svar[var] = float(value) 
                except:
                    break    
    
    else: 
        raise Exception('anal_sim_util.get_sim_info: halo_method not found.')

    if verbose==False:
        return lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum, halo_pos, halo_vel
    else:
        return lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum, halo_pos, halo_vel, halo_svar


def get_sim_info_manual(sim_dir, sim_name, op_idx, method='manual', verbose=False):
    # used for Lya Nature paper  
    if type(op_idx)==int or type(op_idx)==np.int64:
        op_idx = op_idx_int2str(op_idx, 5)
    elif type(op_idx)==str:
        op_idx = op_idx_checkstr(op_idx, 5)
    else:
        raise Exception('op_idx is not the required format')
    
    hydro_file_descriptor = '%s/%s/output_%s/hydro_file_descriptor.txt'%(sim_dir, sim_name, op_idx)
    if verbose==True:
        with open(hydro_file_descriptor, 'r') as f:
            print(f.read())
            
    info_file = '%s/%s/output_%s/info_%s.txt'%(sim_dir, sim_name, op_idx, op_idx)
    lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum = movie_utils.read_infofile(info_file)
    

    if method=='manual':
        # for crazy image
        if sim_dir=='/data/ERCblackholes4/smartin/Spiral1' and sim_name=='RTnsCRiMHD':
            if op_idx=='00029':
                halo_pos = np.array([0.524097, 0.51803,  0.474926])
            elif op_idx == '00033':
                halo_pos = np.array([0.522638, 0.51801,  0.474841]) # exact for 33
            elif op_idx=='00041':
                halo_pos = np.array([0.521297, 0.517836, 0.474855])
            halo_vel = [0,0,0]

    elif method=='manual2':
        # for time evolution
        if sim_dir=='/data/ERCblackholes4/smartin/Spiral1' and sim_name=='RTnsCRiMHD':
            op_idx_a = np.arange(23,42)
            halo_pos_a = np.array( [[0.524467, 0.519456, 0.47476], 
                                    [0.524347, 0.519226, 0.474837], 
                                    [0.524212, 0.519188, 0.474877], 
                                    [0.523982, 0.519006, 0.47492 ], 
                                    [0.52394,  0.51895,  0.474933], 
                                    [0.523756, 0.518771, 0.475078], 
                                    [0.523455, 0.518594, 0.475101], 
                                    [0.523117, 0.518412, 0.475152], 
                                    [0.5228,   0.518159, 0.474978], 
                                    [0.522726, 0.518049, 0.474878], 
                                    [0.522638, 0.51801,  0.474841], 
                                    [0.522513, 0.518058, 0.474766], 
                                    [0.522285, 0.518178, 0.474922], 
                                    [0.522105, 0.518103, 0.47487 ], 
                                    [0.522043, 0.518042, 0.47482 ], 
                                    [0.521832, 0.51783,  0.47481 ], 
                                    [0.521383, 0.517819, 0.474876], 
                                    [0.521297, 0.517836, 0.474855], 
                                    [0.521294, 0.517837, 0.474855]])
            idx = np.where( op_idx_a==int(op_idx) )[0][0]
            halo_pos = halo_pos_a[idx]
            halo_vel = [0,0,0]
    elif method=='manual3':
        # for zoom of 29
        if sim_dir=='/data/ERCblackholes4/smartin/Spiral1' and sim_name=='RTnsCRiMHD':
            if op_idx=='00029':
                halo_pos = np.array([0.52191819, 0.51975412, 0.47431129])
            halo_vel=[0,0,0]

    return lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum, halo_pos, halo_vel

###############################
# read ramses simulation data #
###############################
def define_loading_fields_funct(select):
    # This function defines the proper load fields to upload to memory
    # Use this in combination with the yt.load function
    # ds = yt.load(readf_route, fields=load_fields, extra_particle_fields=extra_particle_fields)
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
    elif (select in ['RTiMHD+SfFb','RTsMHD+SfFb', 'RTiMHD','RTsMHD']):
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


def define_var_name_funct(select):
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


def read_gas_data(dat_dir, sim_suite_dir, sim_name, op_idx, read_var=None, halo_method=None, method='yt', shape='sphere', raw=False):
    '''
    halo_method:  
    method: method for extracting information cube, CDD, yt 
    '''
    lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum, track_pos_halo, track_vel_halo = get_sim_info(sim_suite_dir, sim_name, op_idx, halo_method=halo_method)
    vmean_halo = track_vel_halo*lfac/tfac/1e5 # in km/s

    if method[:3]=='CDD':
        
        directory = '%s/%s/output000%s/'%(dat_dir, sim_name, op_idx )
        cdom = d.domain.read(directory+"compute_domain.dom")
        mdom = m.mesh(filename=directory+"domain_1.mesh")
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
        load_fields = define_loading_fields_funct(sim_name)
        var_names = define_var_name_funct(sim_name)

        size_kpc = 20*u.kpc
        size=size_kpc/(lfac*cm2kpc*u.kpc)
        size=size.decompose().value
        filename = '%s/%s/output_000%s/info_000%s.txt'%(sim_suite_dir, sim_name, op_idx, op_idx)
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
        dvar['vr'] = coord_util.cal_vector_in_r( dvar['x'], dvar['y'], dvar['z'], dvar['vx'], dvar['vy'], dvar['vz'] )

        dvar['p_th'] = dvar['p_th']*dfac*lfac**2/tfac**2 *u.g/u.cm/u.s**2 
        if 'p_cr' in var_names:
            dvar['p_cr'] = dvar['p_cr']*dfac*lfac**2/tfac**2 *u.g/u.cm/u.s**2 
            
        return dvar


######################################################################################
def read_part_data(sim_suite_dir, sim_name, op_idx, size= 2e-3, halo_method='tracker_smartin_dwarf', unit='standard'):
    print('reading particle data')
    if type(op_idx)==int or type(op_idx)==np.int64:
        op_idx = op_idx_int2str(op_idx, 5)
    elif type(op_idx)==str:
        op_idx = op_idx_checkstr(op_idx, 5)
    else:
        raise Exception('op_idx is not the required format')

    lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum, track_pos_halo, track_vel_halo = get_sim_info(sim_suite_dir, sim_name, op_idx, halo_method=halo_method)

    load_fields = define_loading_fields_funct(sim_name)
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

        
        
    # select particle in bbox
    # for some reasons the yt select stars outside of the boundary
    size_kpc = size*Lbox
    idx = np.where( (p_x>-size_kpc/2) & (p_x<size_kpc/2) & \
                    (p_y>-size_kpc/2) & (p_y<size_kpc/2) & \
                    (p_z>-size_kpc/2) & (p_z<size_kpc/2) )[0]
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
        star_age = np.array(star_age)*tH + cosmo.age(z=0).value
        star_age = ( current_time - star_age )*1000 # in Myr
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



