# rascas_util
import os
import sys
sys.path.append('/home/yy503/Desktop/simulation_code/rascas-develop/py')

from cProfile import label
import numpy as np 
import scipy 
from scipy.ndimage import gaussian_filter
import astropy.units as u; import astropy.constants as c
from scipy import stats
from scipy import signal
from scipy.io import FortranFile as ff
from astropy.cosmology import WMAP7

import astro_util
from const_util import *
import anal_sim_util
from astro import obs_util

import jphot as jp 
import lya_utils as ly

# content:
# functions to write block in conf file
# function for generating conf files
# Column density routine
# Analytical calculation of emission line
# read rascas output
# calculate escape fraction

#########################################
# functions to write block in conf file #
#########################################
def write_ramses(f, sim_dir, verbose, deut2H_nb_ratio=3e-5):
    '''
    Check module_ramses.f90 for details
    '''
    f.write('[ramses] \n')         
    sim_name = sim_dir.split('/')[-1]
    if 'RT' in sim_name:
        f.write('self_shielding = F \n')
        f.write('ramses_rt = T \n') 
        f.write('read_rt_variables = T \n')
    else:
        f.write('self_shielding = T \n')
        f.write('ramses_rt = F \n') 
        f.write('read_rt_variables = F \n')
        
    f.write('cosmo = T \n')
    f.write('use_proper_time = T \n') 
            
    f.write('particle_families = F \n')
    f.write('verbose = %s \n'%verbose)
    
    if sim_name=='RTCRiMHD' or sim_name=='RTCRiMHD+SfFb' or sim_name=='RTnsCRiMHD' or sim_name=='RTnsCRiMHD+SfFb':
        f.write('itemp  = 12 \n')
        f.write('imetal = 14 \n')
        f.write('ihii   = 15 \n')
        f.write('iheii  = 16 \n')
        f.write('iheiii = 17 \n')
    elif sim_name=='RT' or sim_name=='RT+SfFb':
        f.write('itemp  =  5\n')
        f.write('imetal =  7\n')
        f.write('ihii   =  8\n')
        f.write('iheii  =  9\n')
        f.write('iheiii =  10\n')
    elif sim_name=='RTiMHD' or sim_name=='RTiMHD+SfFb' or sim_name=='RTsMHD' or sim_name=='RTsMHD+SfFb':
        f.write('itemp  = 11 \n')
        f.write('imetal = 13 \n')
        f.write('ihii   = 14 \n')
        f.write('iheii  = 15 \n')
        f.write('iheiii = 16 \n')
        
    deut2H_nb_ratio = format(deut2H_nb_ratio, '.3e').replace('e','E') 
    f.write('deut2H_nb_ratio   =  %s \n'%deut2H_nb_ratio )
    f.write('recompute_particle_initial_mass = F \n')
    f.write('\n')
    
def write_dust_par(f):
    '''
    Check module_dust.f90 for details
    '''
    f.write('[dust] \n')
    f.write('albedo = 3.2000000000000001e-01 \n')                                        
    f.write('g_dust = 7.2999999999999998e-01 \n')  
    f.write('dust_model = SMC \n') 

def write_gas_composition(f, line, rascas_dir, dust_model, dust_par, overwrite, overwrite_par, verbose):
    '''
    Check module_gas_composition.f90 for details
    '''
    f.write('[gas_composition] \n') 
    if line=='Lya':
        f.write('nscatterer      =  2 \n')
        f.write('scatterer_names = HI-1216, D-1215 \n')
        f.write('atomic_data_dir = %s/ions_parameters/ \n'%rascas_dir)
        f.write('krome_data_dir  = \n')
    elif line in ['LyC', 'LyC_912']:
        pass
    else:
        raise Exception('line %s is not implemented.'%line )
        
    if dust_model=='Laursen09' or dust_model=='ignore':
        f_ion='1.0000000000000000d-02'
        Zref='5.000000000000000d-03'
    elif dust_model=='change_dust':
        f_ion, Zref=dust_par
        f_ion = format(f_ion, '.6e').replace('e','d')
        Zref = format(Zref, '.6e').replace('e','d')
    f.write('f_ion = %s \n'%f_ion )       
    f.write('Zref = %s \n'%Zref)
    f.write('vturb_kms = 0 \n')
    
    if dust_model=='ignore':
        f.write('ignoreDust = T \n')
    else:
        f.write('ignoreDust = F \n')
        
    if overwrite=='F':
        f.write('gas_overwrite = F \n') 
    elif overwrite=='T':
        f.write('gas_overwrite = T \n') 
        fix_nscat, fix_vth_sq_times_m, fix_ndust, fix_vel, fix_vturb, fix_box_size_cm = overwrite_par
        # fix_nscat is a str, other number
        fix_nscat = fix_nscat.replace('e','d') 
        fix_vth_sq_times_m = format(fix_vth_sq_times_m, '.6e').replace('e','d')
        fix_ndust = format(fix_ndust, '.6e').replace('e','d')
        fix_vel = format(fix_vel, '.6e').replace('e','d')
        fix_vturb = format(fix_vturb, '.6e').replace('e','d')
        fix_box_size_cm = format(fix_box_size_cm, '.6e').replace('e','d')
        f.write('fix_nscat   = %s \n'%fix_nscat )
        f.write('fix_vth_sq_times_m   = %s\n'%fix_vth_sq_times_m)
        f.write('fix_ndust   = %s\n'%fix_ndust)
        f.write('fix_vel   = %s\n'%fix_vel)
        f.write('fix_vturb   = %s\n'%fix_vturb)
        f.write('fix_box_size_cm = %s \n'%fix_box_size_cm )
        
    f.write('verbose = %s \n'%verbose)
    f.write('\n')
    
    write_dust_par(f)


def write_ppic_dom(f, dom_type, main_halo_pos, size, pfx ):
    '''
    pfx (prefix) can be 'star' or 'emission'
    '''
    f.write('# computational domain parameters\n')

    f.write('%s_dom_type = %s \n'%(pfx, dom_type) )
    f.write('%s_dom_pos = %f %f %f \n'%(pfx, main_halo_pos[0], main_halo_pos[1], main_halo_pos[2]) )
    # specify size
    if dom_type=='sphere':
        rsp=size/2.
        f.write('%s_dom_rsp = %e \n'%(pfx, rsp) )
    elif dom_type=='cube':
        f.write('%s_dom_size = %e \n'%(pfx, size) )
    elif dom_type=='shell':
        din,dout = size
        rin = din/2 
        rout = dout/2
        f.write('%s_dom_rin = %e \n'%(pfx, rin) )
        f.write('%s_dom_rout = %e \n'%(pfx, rout) )
    elif dom_type=='slab':
        thickness = size 
        f.write('%s_dom_thickness = %e \n'%(pfx, thickness) )
        

######################################
# function for generating conf files # 
######################################
def write_params_CDD(file, dat_dir, sim_dir, snapnum, main_halo_pos, size, cdd_type, line, rascas_dir, dust_model='Laursen09', dust_par=None, overwrite='F',overwrite_par=None, verbose='F', deut2H_nb_ratio=3e-5):
    with open(file,'w') as f:
        f.write('[CreateDomDump] \n')
        f.write('DomDumpDir = %s/ \n'%dat_dir)
        f.write('repository = %s/ \n'%sim_dir)
        f.write('snapnum = %s \n'%snapnum)
        f.write('reading_method = hilbert \n')    
        
        f.write('comput_dom_type = %s \n'%cdd_type) 
        f.write('comput_dom_pos = %f %f %f \n'%(main_halo_pos[0], main_halo_pos[1], main_halo_pos[2]) )
        if cdd_type=='sphere':
            rsp = size/2
            f.write('comput_dom_rsp = %e \n'%rsp )
        elif cdd_type=='cube':
            f.write('comput_dom_size = %e \n'%size )
        
        f.write('decomp_dom_type = %s \n'%cdd_type )
        f.write('decomp_dom_ndomain = 1 \n')
        f.write('decomp_dom_xc = %f \n'%main_halo_pos[0] )
        f.write('decomp_dom_yc = %f \n'%main_halo_pos[1] )
        f.write('decomp_dom_zc = %f \n'%main_halo_pos[2] )
        if cdd_type=='sphere':
            f.write('decomp_dom_rsp = %e \n'%rsp )
        elif cdd_type=='cube':
            f.write('decomp_dom_size = %e \n'%size )
        f.write('verbose = T \n')
        
          
        f.write('[mesh] \n')
        f.write('verbose = T \n')
        
        write_gas_composition(f, line, rascas_dir, dust_model, dust_par, overwrite, overwrite_par, verbose)
        
        write_ramses(f, sim_dir, verbose, deut2H_nb_ratio)


# write ppic files 
def write_params_LyaPhotonFromGas(file, dat_dir, sim_dir, snapnum, main_halo_pos, size, nphotons, rec='False', col='True', emission_dom_type='sphere', verbose='F', IC_tag=None, deut2H_nb_ratio=3e-5):
    f = open(file,'w')
    f.write('[LyaPhotonsFromGas] \n')
    if IC_tag==None:
        if rec=='True':
            f.write('  outputfileRec = %s/rec.IC \n'%dat_dir ) 
        elif col=='True':
            f.write('  outputfileCol = %s/col.IC \n'%dat_dir ) 
    else:
        if rec=='True':
            f.write('  outputfileRec = %s/rec_%s.IC \n'%(dat_dir, IC_tag) ) 
        elif col=='True':
            f.write('  outputfileCol = %s/col_%s.IC \n'%(dat_dir, IC_tag) )     
    f.write('  repository = %s/ \n'%sim_dir )
    f.write('  snapnum = %s \n'%snapnum )
    
    write_ppic_dom(f, emission_dom_type, main_halo_pos, size, 'emission')
        
    if rec=='True':
        f.write('  doRecombs = True \n')
        f.write('  doColls = False \n')
    elif col=='True':
        f.write('  doRecombs = False \n')
        f.write('  doColls = True \n')
    f.write('  tcool_resolution = 5.0e+00 \n') 
    f.write('  nPhotonPackets = %d \n'%nphotons )                    
    f.write('  ranseed = -100 \n' )
    f.write('  verbose = %s \n'%verbose)

    write_ramses(f, sim_dir, verbose, deut2H_nb_ratio)

    f.close()


def write_param_PhotonsFromStars(cfg_file, dat_dir, sim_dir, snapnum, main_halo_pos, size, nphotons, spec_lmin, spec_lmax, filename='star.IC', star_dom_type='sphere', verbose='T', deut2H_nb_ratio=3e-5):
    f = open(cfg_file,'w')
    f.write('[PhotonsFromStars]\n')
    
    f.write('# input / output parameters\n')
    f.write('repository = %s \n'%sim_dir)
    f.write('snapnum    = %s \n'%snapnum )
    f.write('outputfile = %s/%s \n'%(dat_dir, filename) )
    
    write_ppic_dom(f, star_dom_type, main_halo_pos, size, 'star')
                
    f.write('# Spectral shape\n')
    f.write('spec_type               = Table \n')
    f.write('spec_SSPdir        = /data/ERCblackholes2/smartin/Dwarf1/bpassv2_300\n')
    f.write('spec_table_lmin_Ang  =  %.2f \n'%spec_lmin )
    f.write('spec_table_lmax_Ang  =  %.2f \n'%spec_lmax)
    
    f.write('# miscelaneous parameters\n')
    f.write('nPhotonPackets  = %d\n'%nphotons )
    f.write('ranseed         = -100\n' )
    f.write('verbose         = %s\n'%verbose)

    write_ramses(f, sim_dir, verbose, deut2H_nb_ratio)

    f.close()

# write RASCAS configuration file 
def write_params_Ra(file, dat_dir, PhotICFile, outputFile, ndir, mock_name, line, rascas_dir, dust_model='Laursen09', dust_par=None, overwrite='F', overwrite_par=None, verbose='T', verbose_worker='T', nbundle=10, mock_parameter_file='mockparams.conf' ):
    f = open(file,'w')
    f.write('[RASCAS]\n')
    f.write('DomDumpDir = %s/ \n'%dat_dir)
    f.write('PhotonICFile = %s/%s\n'%(dat_dir, PhotICFile))
    f.write('fileout      = %s/%s\n'%(dat_dir, outputFile))
    f.write('nbundle = %d \n'%nbundle)
    f.write('verbose      = %s \n'%verbose )

    if ndir!=0:                                                   
        f.write('[mock] \n')
        f.write('nDirections = %d \n'%ndir)                                                                    
        f.write('mock_parameter_file = %s/%s \n'%(dat_dir, mock_parameter_file) )   
        f.write('mock_outputfilename = %s/%s \n'%(dat_dir, mock_name) )
    
    f.write('[master]\n') 
    f.write('verbose = %s\n'%verbose)
    f.write('restart = F \n')
    f.write('PhotonBakFile  = %s/%s.backup \n'%(dat_dir, outputFile) )
    f.write('dt_backup      = 3600 \n')

    f.write('[worker] \n') 
    f.write('verbose = %s \n'%verbose_worker ) 
    
    if line=='Lya':
        write_gas_composition(f, line, rascas_dir, dust_model, dust_par, overwrite, overwrite_par, verbose)
        
        f.write('[HI-1216] \n')
        f.write('recoil = T \n')
        f.write('isotropic = F \n') 
        f.write('core_skip = T \n')                                                       
        f.write('xcritmax = 1000.0e+00 \n') 
        
    elif line=='LyC_912':
        no_scatter     = 'F'          
        use_dust       = 'T'          
        use_helium     = 'T'          
        use_v          = 'T'          
        analytic       = 'T'          
        f.write('[photon] \n')
        f.write('  no_scatter      = %s \n \n'%no_scatter)
        f.write('[gas_composition] \n')
        f.write('  use_dust        = %s \n'%use_dust)
        f.write('  use_helium     = %s \n'%use_helium)
        f.write('  use_v           = %s \n'%use_v)
        f.write('[lines] \n')
        f.write('  analytic       = %s \n'%analytic)
        
    else:
        raise Exception('line %s is not implemented.'%line )
      
    write_dust_par(f)

    f.write('[uparallel] \n')
    f.write('method        = RASCAS \n')
    f.write('xForGaussian  = 8. \n')
    
    f.write('[voigt] \n')
    f.write('approximation = COLT \n')

    f.close()

def write_mockparams(file, LOS_a, main_halo_pos, flux_aper, 
                    spec_npix, spec_aperture, spec_lmin, spec_lmax,
                    image_npix, image_side,
                    cube_lbda_npix, cube_image_npix, cube_lmin, cube_lmax, cube_side ):
    with open(file, 'w') as f:
        for LOS in LOS_a:
            f.write('%f %f %f \n'%(LOS[0], LOS[1], LOS[2]) ) 
            f.write('%f %f %f \n'%(main_halo_pos[0], main_halo_pos[1], main_halo_pos[2] ) ) 
            f.write('%f \n'%flux_aper)  
            f.write('%d %f %f %f\n'%(spec_npix, spec_aperture, spec_lmin, spec_lmax) ) 
            f.write('%d %f \n'%(image_npix, image_side) )    
            f.write('%d %d %f %f %f \n'%(cube_lbda_npix, cube_image_npix, cube_lmin, cube_lmax, cube_side) )


def write_mockparams_new_ions(file, LOS_a, main_halo_pos, flux_aper, 
                    spec_npix, spec_aperture, spec_lmin, spec_lmax,
                    image_npix, image_side,
                    cube_lbda_npix, cube_image_npix, cube_lmin, cube_lmax, cube_side,
                    rings_lbda_npix, rings_npix, rings_lmin, rings_lmax, rings_aperture):
    with open(file, 'w') as f:
        for LOS in LOS_a:
            f.write('%f %f %f \n'%(LOS[0], LOS[1], LOS[2]) ) 
            f.write('%f %f %f \n'%(main_halo_pos[0], main_halo_pos[1], main_halo_pos[2] ) ) 
            f.write('%f \n'%flux_aper)  
            f.write('%d %f %f %f\n'%(spec_npix, spec_aperture, spec_lmin, spec_lmax) ) 
            f.write('%d %f \n'%(image_npix, image_side) )    
            f.write('%d %d %f %f %f \n'%(cube_lbda_npix, cube_image_npix, cube_lmin, cube_lmax, cube_side) )
            f.write('%d, %d, %f, %f, %f \n'%(rings_lbda_npix, rings_npix, rings_lmin, rings_lmax, rings_aperture) )
            
##########################
# Column density routine #
##########################
def write_params_IC_CD(file, IC_CD_par, star_dom_par, sim_dir):
    '''
    Check IC_ColumnDensity.f90 for details
    method can be fromstars, onepoint, npoints
    '''
    method=IC_CD_par['method']
    with open(file,'w') as f:
        f.write('[IC_ColumnDensity] \n')
        f.write('outputfile = %s \n'%IC_CD_par['outputfile'] )
        f.write('repository = %s \n'%IC_CD_par['repository'] )
        f.write('snapnum = %s \n'%IC_CD_par['snapnum'] )        
        f.write('method = %s \n'%method )

        if method=='onepoint':
            x_phot = IC_CD_par['x_phot']
            f.write('x_phot = %f %f %f \n'%(x_phot[0], x_phot[1], x_phot[2]) )
        
        if method=='npoints':
            f.write('pos_parameter_file = %s \n'%IC_CD_par['pos_parameter_file'] )
            f.write('nphot = %s \n'%IC_CD_par['nphot'] )
            
        if method=='fromstars':
            star_dom_type = star_dom_par['star_dom_type']
            main_halo_pos = star_dom_par['main_halo_pos']
            size = star_dom_par['size']
            write_ppic_dom(f, star_dom_type, main_halo_pos, size, 'star')
        
        f.write('verbose = T \n')
        
        write_ramses(f, sim_dir, verbose='T', deut2H_nb_ratio=3e-5)
        
def write_pos_parameter_file(file, nphot, xvec_em ):
    '''
    Write pos_parameter_file for IC_CD
    '''
    with open(file, 'w') as f:
        for i in range(nphot):
            x_em, y_em, z_em = xvec_em[i]
            f.write('%f %f %f \n'%( x_em, y_em, z_em ) )
    
        
def write_params_CD(file, CD_par, gas_comp_par):
    with open(file,'w') as f:
        f.write('[ColumnDensity] \n')
        f.write('verbose = %s \n'%CD_par['verbose'] )
        f.write('DomDumpDir = %s/ \n'%CD_par['DomDumpDir'] )
        f.write('RaysICFile = %s \n'%CD_par['RaysICFile'] )
        f.write('path_IC = %s \n'%CD_par['path_IC'] )
        f.write('n_IC = %d \n'%CD_par['n_IC'] )   # number of IC files, they are stored in dir_1 to dir_n_IC folder
        f.write('nDirections = %d \n'%CD_par['nDirections'] )
        f.write('direction_file = %s \n'%CD_par['direction_file'] )
        f.write('fileout = %s \n'%CD_par['fileout'] )
        
        f.write('[mesh] \n')
        f.write('verbose = T \n')
        
        write_gas_composition(f, gas_comp_par['line'], gas_comp_par['rascas_dir'], gas_comp_par['dust_model'], gas_comp_par['dust_par'], gas_comp_par['overwrite'], gas_comp_par['overwrite_par'], gas_comp_par['verbose'])
        
        
###########################################
# Analytical calculation of emission line #
###########################################
kB = ly.kb_cgs
e_lya = ly.h_cgs*ly.clight/ly.lambda0 *u.erg

def cal_Lya_rec(T, ne, nHII):
    # calculate Lya from recombination
    # Cantalupo+(08)
    if type(T)== u.quantity.Quantity:
        TK = (T/u.K).decompose().value
    
    Ta = np.maximum(TK, 100.0 ) # no extrapolation..
    prob_case_B = 0.686 - 0.106*np.log10(Ta/1e4) - 0.009*(Ta/1e4)**(-0.44)
    # Hui & Gnedin (1997)
    lambd = 315614/TK
    alpha_B = 2.753e-14*(lambd**(1.5))/(1+(lambd/2.74)**0.407)**(2.242) *u.cm**3/u.s 
    Lya_rec = prob_case_B * alpha_B * ne * nHII * e_lya # [erg/cm3/s]
    Lya_rec = Lya_rec.to(u.erg/u.cm**3/u.s)
    
    return prob_case_B, alpha_B, Lya_rec

def cal_Lya_col(T, nHI, ne, coeff='Goerdt10'):
    # calculate Lya from collisional excitation
    if type(T)== u.quantity.Quantity:
        TK = (T/u.K).decompose().value
        
    if coeff=='Goerdt10':
        # eq 10 in Goerdt+10 
        collExrate_HI = 2.41e-6/np.sqrt(TK) * (TK/1.e4)**0.22 * np.exp(-1.63e-11/(kB*TK)) *u.cm**3/u.s 
        Lya_col = nHI * ne * collExrate_HI * e_lya 

    elif coeff=='Harley':
        # eq (A3) and table 1 in Katz+22 MgII paper
        collExrate_HI = (6.58e-18 / TK**0.185) * np.exp(-4.86e4/TK**0.895) *u.erg*u.cm**3/u.s 
        Lya_col = nHI * ne * collExrate_HI  
        
    Lya_col = Lya_col.to(u.erg/u.cm**3/u.s)
        
    return collExrate_HI, Lya_col

def cal_line_col():
    # TBW
    # eq (A3) and table 1 in Katz+22 MgII paper
    def col_emiss(T_e, a,b,c,d):
        eps = a/T_e**c * np.exp(-b/T_e**d) *u.erg*u.cm**3/u.s 
    

######################
# read rascas output #
######################
def rascas_read_info(file_fd, lfac):
    '''
    Read mockparams
    '''
    with open('%s/mockparams.conf'%file_fd, 'r') as f:
        f.readline() 
        f.readline()
        aper = float(f.readline() )
        spec_npix, _, spec_lmin, spec_lmax = [float(x) for x in f.readline().split()]
        spec_npix = int(spec_npix)
        image_npix, _ = [float(x) for x in f.readline().split()]
        image_npix = int(image_npix)
        f.readline()
    aper = aper*lfac*cm2kpc 
    return aper, spec_npix, spec_lmin, spec_lmax, image_npix

def rascas_Lumpp(file_fd, mech):
    p = jp.photonlist('%s/%s.IC'%(file_fd, mech), None)
    nRealPhotons = p.nRealPhotons
    nPhotons     = p.nphoton
    nPhotPerPacket = nRealPhotons / nPhotons

    if 'rec' in mech or 'col' in mech or mech=='star':
        LumPerPacket = nPhotPerPacket * ly.h_cgs * ly.nu0 # this assumes a narrow Lambda range (ie all photons have frequency nu0)
    elif mech=='LyC':
        LumPerPacket = nPhotPerPacket * ly.h_cgs * ly.clight / (912e-8)  # very rough approximation, for LyC photon near 912
    return nPhotons, nPhotPerPacket, LumPerPacket

def unit_conv(file_fd, mech, spec_lmin, spec_lmax, spec_npix, image_npix, z, size, frame='rest'):
    nPhotons, nPhotPerPacket, LumPerPacket = rascas_Lumpp(file_fd, mech)
    #SB_unit = LumPerPacket/(size*kpc2cm/image_npix)**2/(4*np.pi*(1+z)**4 )/Sr2arcsec2 # [erg / s / cm2 / arcsec2 / phot packet]
    dx = size*u.kpc/image_npix
    SB_unit = obs_util.Lpix2SB(LumPerPacket*u.erg/u.s, dx, z).value

    # restframe spectrum
    if frame=='rest':
        dl = (spec_lmax - spec_lmin) / spec_npix  # [A] 
        l = np.arange(spec_lmin,spec_lmax,dl) + 0.5*dl # [A]
    LumPerPac_l = nPhotPerPacket * ly.h_cgs * ly.clight / (l*1e-8)  # [erg / s / phot packet]

    #spct_unit = LumPerPac_l/(1+z)**4/(size*kpc2cm)**2/Sr2arcsec2  / dl   # [erg / s / cm2 /  arcsec2 / A / phot packet] 
    #ppv_unit = LumPerPac_l/(1+z)**4/(size*kpc2cm/image_npix)**2/Sr2arcsec2  / dl   # [erg / s / cm2 /  arcsec2 / A / phot packet] 
    spct_unit = obs_util.Lppv2spec_int2(LumPerPac_l*u.erg/u.s, size*u.kpc, dl*u.Angstrom, z).value
    ppv_unit = obs_util.Lppv2spec_int2(LumPerPac_l*u.erg/u.s, dx, dl*u.Angstrom, z).value

    return nPhotons, LumPerPacket, SB_unit, spct_unit, ppv_unit

def read_flux(filepath, ndir):
    flux_a = []
    with ff(filepath) as file:
        for i in range(ndir):
            arr = file.read_reals(dtype='f8') 

            if len(arr)==3:
                aper, flux, flux_hnu = arr
            elif len(arr)==2:
                aper, flux = arr
            
            flux_a.append( flux )
    flux_a = np.array(flux_a)
    
    return flux_a

def read_spectrum(filepath, ndir):
    spectrum_a = []
    with ff(filepath) as file:
        for i in range(ndir):
            spec_npix = file.read_ints()[0]
            aper, spec_lmin, spec_lmax = file.read_reals(dtype='f8')
            spectrum = file.read_reals(dtype='f8')
            spectrum_a.append( spectrum )
    spectrum_a = np.array(spectrum_a)
    return spec_lmin, spec_lmax, spec_npix, spectrum_a

def read_image(filepath, ndir):
    image_a = []
    with ff(filepath) as file:
        for i in range(ndir):
            image_npix = file.read_ints()[0]
            size = file.read_reals(dtype='f8')[0]
            center = file.read_reals(dtype='f8')
            image = file.read_reals(dtype='f8').reshape(image_npix, image_npix)
            if 'LyC' in filepath:
                image_hnu= None
            else:
                image_hnu= file.read_reals(dtype='f8')
            image_a.append( image )
    image_a = np.array(image_a)
    return center, size, image_npix, image_a, image_hnu

def read_cube(filepath, ndir):
    file = ff(filepath)
    cube_a = []
    for i in range(ndir):
        spec_npix, image_npix  = file.read_ints()
        file.read_reals(dtype='f8')
        file.read_reals(dtype='f8')
        cube = file.read_reals(dtype='f8').reshape(image_npix, image_npix, spec_npix)
        cube_a.append( cube )
    cube_a = np.array(cube_a)

    return cube_a

def read_rascas_output(file_fd, redshiftnum, lfac, mech, noise_floor=True):
    # mech: col, rec, intrcol, intrrec
    print('read_rascas_output: reading', mech)
    LOS_a = np.loadtxt('%s/LOS.txt'%file_fd)

    if LOS_a.ndim==2:
        ndir = np.shape( LOS_a )[0]
    elif LOS_a.ndim==1:
        ndir = 1

    if 'LyC' in mech:
        try:
            flux_a = read_flux('%s/%s.res_flux'%(file_fd, mech), ndir)
            spec_lmin, spec_lmax, spec_npix, spectrum_a = read_spectrum('%s/%s.res_spectrum'%(file_fd, mech), ndir)
            center, size, image_npix, image_a, _ = read_image('%s/%s.res_image'%(file_fd, mech), ndir)
        except Exception as e:
            print(e)
            flux_a = 0
            spec_lmin, spec_lmax, spec_npix, spectrum_a = 0,0,0,0
            center, size, image_npix, image_a = 0,0,0,0
        try:
            cube_a = read_cube('%s/%s.cube'%(file_fd, mech), ndir)
        except:
            print('cannot read cube')
            cube_a = None
    else:
        try:
            flux_a = read_flux('%s/%s.flux'%(file_fd, mech), ndir)
            spec_lmin, spec_lmax, spec_npix, spectrum_a = read_spectrum('%s/%s.spectrum'%(file_fd, mech), ndir)
            center, size, image_npix, image_a, image_hnu = read_image('%s/%s.image'%(file_fd, mech), ndir)
        except Exception as e:
            print(e)
            flux_a = 0
            spec_lmin, spec_lmax, spec_npix, spectrum_a = 0,0,0,0
            center, size, image_npix, image_a = 0,0,0,0    
        try:
            cube_a = read_cube('%s/%s.cube'%(file_fd, mech), ndir)
        except:
            print('cannot read cube')
            cube_a = None

    # unit conversion
    size = size*lfac*cm2kpc

    if 'intr' in mech or 'zoom' in mech:
        mech2 = mech[4:]
    else:
        mech2 = mech

    # Poisson noise (in counts)
    noise_flux_a = np.sqrt(flux_a)
    noise_spectrum_a = np.sqrt(spectrum_a)
    noise_image_a = np.sqrt(image_a)
    if cube_a is not None:
        noise_cube_a = np.sqrt(cube_a)
    else:
        noise_cube_a = None 

    # noise floor
    if noise_floor:
        floor = 1
        noise_flux_a[noise_flux_a<floor] = floor
        noise_spectrum_a[noise_spectrum_a<floor] = floor
        noise_image_a[noise_image_a<floor] = floor
        if noise_cube_a is not None:
            noise_cube_a[noise_cube_a<floor] = floor

    # add unit 
    nPhotons, LumPerPacket, SB_unit, spct_unit, ppv_unit = unit_conv(file_fd, mech2, spec_lmin, spec_lmax, spec_npix, image_npix, redshiftnum, size)
    flux_a = flux_a * LumPerPacket
    spectrum_a = spectrum_a * np.reshape( spct_unit, (1, spct_unit.size) )
    image_a = image_a * SB_unit
    if cube_a is not None:
        cube_a = cube_a * np.reshape( ppv_unit, (1, ppv_unit.size) )
    
    noise_flux_a *= LumPerPacket
    noise_spectrum_a *= np.reshape( spct_unit, (1, spct_unit.size) )
    noise_image_a *= SB_unit
    if noise_cube_a is not None:
        noise_cube_a  *= np.reshape( ppv_unit, (1, ppv_unit.size) )

    return size, center, spec_lmin, spec_lmax, spec_npix, image_npix, ndir, nPhotons, LumPerPacket, SB_unit, spct_unit, ppv_unit, flux_a, spectrum_a, image_a, cube_a, noise_flux_a, noise_spectrum_a, noise_image_a, noise_cube_a


def read_Lya_aavg(emi_dir, ppic_method='sphere'):
    '''
    Read Lya output
    '''
    print('read_Lya aavg')
    dvar = {}
    
    if ppic_method=='sphere':
        mech_a=['col', 'rec']
    elif ppic_method=='twoshell':
        mech_a=['col_ism', 'col_cgm', 'rec_ism', 'rec_cgm']
    elif ppic_method=='star':
        mech_a = ['star']
        
    for mech in mech_a:
        dvar['p_%s'%mech] = jp.photonlist('%s/%s.IC'%(emi_dir, mech), '%s/%s.res'%(emi_dir, mech),load=True)  

    #spec_lmin = 
    #spec_lmax = 
    #spec_npix = 
    #lam = np.linspace(spec_lmin, spec_lmax, spec_npix)
    #v = astro_util.lam_to_v(lam, lam_lya)

    intr_flux = 0
    aavg_flux = 0
    #spectrum_aavg = 0
    #intr_spectrum_aavg = 0
    for mech in mech_a:
        
        intr_flux += dvar['p_%s'%mech].flux(frame='ic')
        aavg_flux += dvar['p_%s'%mech].flux(frame='obs')
        #intr_spectrum_aavg += dvar['p_%s'%mech].spectrum(frame='ic',nbins=spec_npix,unit='count',lmin=spec_lmin, lmax=spec_lmax)[1]*dvar['specunit_%s'%mech]
        #spectrum_aavg += dvar['p_%s'%mech].spectrum(frame='obs',nbins=spec_npix,unit='count',lmin=spec_lmin, lmax=spec_lmax)[1]*dvar['specunit_%s'%mech]
   
    fesc_lya_aavg = aavg_flux/intr_flux
    
    return intr_flux, aavg_flux, fesc_lya_aavg


def read_Lya(emi_dir, redshiftnum, lfac, ppic_method='sphere', more_op=False, noise_floor=True):
    '''
    Read Lya output
    '''
    print('read_Lya')
    dvar = {}
    LOS_a = np.loadtxt('%s/LOS.txt'%emi_dir)
    if LOS_a.ndim==1:
        LOS_a = np.array([LOS_a])
    
    if ppic_method=='sphere':
        mech_a=['col', 'rec']
    elif ppic_method=='twoshell':
        mech_a=['col_ism', 'col_cgm', 'rec_ism', 'rec_cgm']
    elif ppic_method=='star':
        mech_a = ['star']
        
    for mech in mech_a:
        dvar['p_%s'%mech] = jp.photonlist('%s/%s.IC'%(emi_dir, mech), '%s/%s.res'%(emi_dir, mech),load=True)  

        size, center, dvar['spec_lmin_%s'%mech], dvar['spec_lmax_%s'%mech], dvar['spec_npix_%s'%mech], image_npix, ndir, nPhotons, dvar['LpPac_%s'%mech ], dvar['SBunit_%s'%mech ], dvar['specunit_%s'%mech ], dvar['ppv_unit_%s'%mech], \
        dvar['%s_flux_a'%mech], dvar['%s_spectrum_a'%mech], dvar['%s_image_a'%mech], dvar['%s_cube_a'%mech], \
        dvar['%s_noise_flux_a'%mech], dvar['%s_noise_spectrum_a'%mech], dvar['%s_noise_image_a'%mech], dvar['%s_noise_cube_a'%mech] = \
        read_rascas_output(emi_dir, redshiftnum, lfac, mech, noise_floor)

        nbins=dvar['spec_npix_%s'%mech]
        _, dvar['intr_spectrum_%s_a'%mech] = dvar['p_%s'%mech].intrinsic_spectrum(nbins=nbins,Flambda=False,lmin=dvar['spec_lmin_%s'%mech],lmax=dvar['spec_lmax_%s'%mech], kobs_a=LOS_a )
        
        # intrinsic
        dvar['intr_spectrum_%s_a'%mech] *= dvar['specunit_%s'%mech].reshape( (1, nbins) )
        dvar['intr_image_%s_a'%mech] = dvar['p_%s'%mech].mock_intrinsic_image(center, kobs_a=LOS_a, size=size/(lfac*cm2kpc), image_npix=image_npix, unit='count' )*dvar['SBunit_%s'%mech]
        dvar['intr_ppv_%s_a'] = dvar['p_%s'%mech].mock_intrinsic_ppv(center, kobs_a=LOS_a, ppv_range=( size/(lfac*cm2kpc), dvar['spec_lmin_%s'%mech], dvar['spec_lmax_%s'%mech] ), ppv_res=(image_npix, dvar['spec_npix_%s'%mech] ), unit='count')*dvar['ppv_unit_%s'%mech]
        
        if mech=='star':
            # deal with boundary effect
            dvar['%s_spectrum_a'%mech][:, 0] = 0
            dvar['%s_spectrum_a'%mech][:, -1] = 0
            

    spec_lmin = dvar['spec_lmin_%s'%mech_a[0]]
    spec_lmax = dvar['spec_lmax_%s'%mech_a[0]]
    spec_npix = dvar['spec_npix_%s'%mech_a[0]]
    lam = np.linspace(spec_lmin, spec_lmax, spec_npix)
    lam_star = None
    v = astro_util.lam_to_v(lam, lam_lya)

    flux_a = 0
    spectrum_a=0
    image_a = 0
    cube_a=0
    noise_flux_a = 0
    noise_spectrum_a = 0
    noise_image_a = 0
    noise_cube_a = 0
    intr_spectrum_a = 0
    intr_image_a = 0
    intr_ppv_a = 0
    intr_flux = 0
    aavg_flux = 0
    spectrum_aavg = 0
    intr_spectrum_aavg = 0
    for mech in mech_a:
        flux_a += dvar['%s_flux_a'%mech]
        spectrum_a += dvar['%s_spectrum_a'%mech]
        image_a += dvar['%s_image_a'%mech]
        cube_a += dvar['%s_cube_a'%mech]
        noise_flux_a += dvar['%s_noise_flux_a'%mech]
        noise_spectrum_a += dvar['%s_noise_spectrum_a'%mech]
        noise_image_a += dvar['%s_noise_image_a'%mech]
        noise_cube_a += dvar['%s_noise_cube_a'%mech]
        
        intr_spectrum_a += dvar['intr_spectrum_%s_a'%mech] 
        intr_image_a += dvar['intr_image_%s_a'%mech]
        intr_ppv_a += dvar['intr_ppv_%s_a']
        
        intr_flux += dvar['p_%s'%mech].flux(frame='ic')
        aavg_flux += dvar['p_%s'%mech].flux(frame='obs')
        intr_spectrum_aavg += dvar['p_%s'%mech].spectrum(frame='ic',nbins=spec_npix,unit='count',lmin=spec_lmin, lmax=spec_lmax)[1]*dvar['specunit_%s'%mech]

        spectrum_aavg += dvar['p_%s'%mech].spectrum(frame='obs',nbins=spec_npix,unit='count',lmin=spec_lmin, lmax=spec_lmax)[1]*dvar['specunit_%s'%mech]
   
    # orientation problem
    intr_image_a = np.flip(intr_image_a, axis=2)
    intr_ppv_a = np.flip(intr_ppv_a, axis=2)

    image_a = np.rot90(image_a, axes=(2,1) )
    cube_a = np.rot90(cube_a, axes=(2,1) )    
    noise_image_a = np.rot90( noise_image_a, axes=(2,1) )
    noise_cube_a = np.rot90( noise_cube_a, axes=(2,1) )

    fesc_lya_aavg = aavg_flux/intr_flux
    fesc_lya_a = flux_a/intr_flux 
    
    if more_op==False:
        return size, ndir, LOS_a, lam, v, lam_star, flux_a, spectrum_a, image_a, cube_a, intr_spectrum_a, intr_image_a, intr_ppv_a, intr_flux, fesc_lya_aavg, fesc_lya_a
    elif more_op==True:
        return size, ndir, LOS_a, lam, v, lam_star, flux_a, spectrum_a, image_a, cube_a, intr_spectrum_a, intr_image_a, intr_ppv_a, intr_flux, fesc_lya_aavg, fesc_lya_a, intr_spectrum_aavg, spectrum_aavg, noise_flux_a, noise_spectrum_a, noise_image_a, noise_cube_a


def read_CD_output(path_CD_dir, gas_mix='HI_DI_dust'):
    '''
    Different gas species (all elements + dust) are in different files. 
    Each files first gives # of rays, # of directions, # of gas species. Then the file gives 2d histo as ndir, nrays for particular gas.
    Check subroutine dump in module_CD.f90 in rascas-new_ions for details.
    '''
    filepath = '%s/CD.out_01'%(path_CD_dir) 
    with ff(filepath) as file:
        nrays = file.read_ints()[0]
        ndir = file.read_ints()[0]
        nGas = file.read_ints()[0]
        
    histo_a = np.zeros( (nGas, ndir, nrays) )
    for i in range(nGas):
        iop = i+1
        i_str = anal_sim_util.op_idx_int2str(iop, L=2)
        filepath = '%s/CD.out_%s'%(path_CD_dir, i_str) 
                
        with ff(filepath) as file:
            file.read_ints()
            file.read_ints()
            file.read_ints()

            for j in range(ndir):
                histo_a[i,j,:] = file.read_reals(dtype='f8')

    return nrays, ndir, histo_a

#############################
# calculate escape fraction #
#############################
def cal_fesc_spec(spectrum_a, intr_flux, redshiftnum, size, spec_lmin, spec_lmax, spec_npix, frame='rest'):
    '''
    Calculate fesc from spectrum
    '''
    ndir = np.shape(spectrum_a)[0]
    fesc_a = np.zeros(ndir)
    
    if frame=='redshifted':
        spec_lmin *= 1+redshiftnum 
        spec_lmax *= 1+redshiftnum
    elif frame=='rest':
        pass 

    for i, spectrum in enumerate(spectrum_a):
        dl = (spec_lmax - spec_lmin) / spec_npix 
        fac = 4*np.pi*(1+redshiftnum)**4 * (size*kpc2cm)**2 * Sr2arcsec2 * dl
        Ltot = np.sum( spectrum )*fac
        fesc_a[i] = Ltot/intr_flux
    return fesc_a

def cal_fesc_obs_image(image_a, noise, npix_tele, beam_size, redshift, size, intr_flux):
    '''
    This function calculate observed flux
    This take into account (1) noise effect (2) resolution effect (3) beam smearing effect 
    '''

    lpix = size/npix_tele
    fac = 4*np.pi*(1+redshift)**4*lpix**2*Sr2arcsec2
    if image_a.ndim==2:
        image = image_a
        mock_img = astro_util.make_tele_image(image, noise, npix_tele, beam_size)
        sum_SB = np.sum(mock_img)        
        flux = sum_SB*fac
        fesc = flux/intr_flux
        return fesc

    elif image_a.ndim==3:
        ndir = np.shape(image_a)[0]
        fesc_a = np.zeros(ndir)
        for i, image in enumerate(image_a):
            mock_img = astro_util.make_tele_image(image, noise, npix_tele, beam_size)
            sum_SB = np.sum(mock_img)        
            flux = sum_SB*fac
            fesc_a[i] = flux/intr_flux
        return fesc_a
    
def cal_fesc_obs_ppv(ppv, noise, nspec_tele, npix_tele, beam_size, intr_flux ):
    mock_ppv = astro_util.make_tele_ppv(ppv, noise, nspec_tele, npix_tele, beam_size)
    sum_ppv = np.sum(mock_ppv)
    # TBW, calculate flux from sum_ppv
    flux=np.nan
    fesc = flux/intr_flux
    return fesc



