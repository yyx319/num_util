import os
import sys
from cProfile import label
from lib2to3.pygram import python_grammar_no_print_statement
from multiprocessing import Pool

import re
import numpy as np 
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import scipy 
from scipy.ndimage import gaussian_filter

import yt

sys.path.append('/data/ERCblackholes3/yuxuan')
import astropy.units as u; import astropy.constants as c
from scipy import stats
sys.path.append('/home/yy503/Desktop/simulation_code/ramses_tools/HaloMakerRoutines/')
import NewHaloMakerRoutines as hmr
sys.path.append('/home/yy503/Desktop/simulation_code/ramses_tools/MakeMovies')
import movie_utils
from support_functions import *
from scipy.interpolate import interp1d
from scipy import signal
from scipy.io import FortranFile as ff

from astropy.cosmology import WMAP7

import anal_sim_util 
import astro_util
import coord_util
from const_util import *



'''
Content:
'''

###
# polish the plot
#
def sci_fm(x):
    '''
    change x to scientific format (for publication)
    '''
    if type(x) is str:
        x = float(x)
    
    a,b = ('%.2e'%x).split('e')
    
    # formating exponent
    if len(b)==3 and b[:2]=='+0': 
        b = b[2]
    elif len(b)==3 and b[:2]=='-0':
        b = b[0]+b[2]
    else:
        pass 
    
    x_sci_fm = '$'+a+'\\cdot 10^{'+b+'}$' 
    return x_sci_fm 


##################
# Plot templates #
##################

def scatter_hist(x1, y1, ax, ax_histx, ax_histy, bins_x, bins_y, plot='hist'):
    #https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    if plot=='hist':
        ax.hist2d(x1, y1, bins=[bins_x, bins_y])
    elif plot=='scatter':
        ax.scatter(x1, y1, s=3)

    # now determine nice limits by hand:
    ax_histx.hist(x1, bins=bins_x, histtype='step')
    ax_histy.hist(y1, bins=bins_y, orientation='horizontal', histtype='step')

##############################
# function for color of plot # 
##############################

def gen_color_arr(n, cmap='Blues'):
    '''
    generate array representing different depth of with the same color
    '''
    norm = colors.Normalize(vmin=-0.5, vmax=1)
    cmap1 = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap( cmap ) )
    #plt.rcParams.
    rg = range(n)
    rgb_a = []
    for i in rg:
        rg2 = float( i-min(rg) )/(len(rg)-1) 
        rgb_a.append( cmap1.to_rgba(rg2) )
    return rgb_a

def gen_misc_color_arr(n, method='default'):
    if method=='default':
        color_a = plt.rcParams['axes.prop_cycle'].by_key()['color'][:n]
    return color_a 
    
    
'''
Construct the frame of the plot
'''
def cal_fig_size(nrow, ncol, wspace, hspace, boundary, panel_ratio=1, mode='publication'):
    '''
    Calculate the figure size
    '''
    if type(boundary) in [list, tuple, np.ndarray]:
        bottom, top, left, right = boundary
    
    if panel_ratio=='for_healpix':
        panel_ratio=0.5
    elif panel_ratio=='for_image':
        panel_ratio=1

    # set figure width
    if mode in ['pub', 'publication']:
        fig_w = 25 
    elif mode in ['pub_single_col', 'publication_single_col']:
        fig_w = 12.5
    elif mode=='normal':
        fig_w = ncol*3
    
    panel_w = fig_w * ( right - left ) / (ncol + (ncol-1)*wspace )  # panel width in inches
    panel_h = panel_w*panel_ratio # panel height in inches
    fig_h = (nrow + (nrow-1)*hspace ) * panel_h / ( top-bottom )  # in inches
    
    return fig_w, fig_h

calculate_fig_size = cal_fig_size

def colorbar_edges(nrow, bottom, top, cb_frac, hspace, orientation='vertical'):
    '''
    generate the lower edges of color bars from top to bottom 
    nrow: number of rows 
    bottom, top: bottom and top of the figure
    cb_frac: fraction of colorbar height 
    hspace: same parameters in plt.subplots()
    '''
    panel_h = (top-bottom) / (nrow + (nrow-1)*hspace ) 
    if orientation=='vertical':
        lcb = panel_h*cb_frac  # length of colorbar
        d = panel_h*(1-cb_frac)/2 # vertical distance between colorbar bottom (top) edge to panel bottom (top) edge
        cb_edge_a = np.linspace(top-panel_h+d, bottom+d, nrow) # colorbar lower edges
        return cb_edge_a, lcb
    
    elif orientation=='horizontal':
        wcb = panel_h*hspace*cb_frac 
        d = panel_h*hspace*(1-cb_frac)
        d1 = d*0.1
        d2 = d*0.9
        cb_edge_a = np.linspace( top-panel_h-d1-wcb, bottom-d1-wcb, nrow)
        return cb_edge_a, wcb

colorbar_edges_t2b = colorbar_edges

def colorbar_edges_l2r():
    pass 

def colorbar_edges_all_panel():
    pass


##################################
# post processing the simulation # 
##################################
def general_1d_hist(var_a, var_bin, field_a, weighted_field_a=None, stat='mean', smooth=False):
    if var_a is not None and var_a.ndim>=2:
        var_a = var_a.flatten()
    if field_a is not None and field_a.ndim>=2:
        field_a = field_a.flatten()
    if weighted_field_a is not None and weighted_field_a.ndim>2:
        weighted_field_a = weighted_field_a.flatten()
        
    if weighted_field_a is None:
        f_var, _, _ = stats.binned_statistic(var_a, field_a, stat, var_bin)
    else:
        fwf_r, _, _ = stats.binned_statistic(var_a, field_a*weighted_field_a, stat, var_bin)
        wf_r, _, _ = stats.binned_statistic(var_a, weighted_field_a, stat, var_bin)
        f_var = fwf_r/wf_r 
    var_a2 = 1/2*( var_bin[1:] + var_bin[:-1] )
    return var_a2, f_var  

radial_profile = general_1d_hist 

def general_2d_histogram(cube_field1, cube_field2, cube_field3, f1_bin=None, f2_bin=None, cube_weight_field=None, stat='sum'):
    '''
    x y coordinate field 1, 2; value taken as field 3, weighted field is optional
    '''

    if cube_field1.ndim>=2:
        cube_field1 = cube_field1.flatten()
        cube_field2 = cube_field2.flatten()
        cube_field3 = cube_field3.flatten()
    
    if f1_bin is None:
        f1_bin = np.linspace( np.min(cube_field1), np.max(cube_field1), 100 )
    if f2_bin is None:
        f2_bin = np.linspace( np.min(cube_field2), np.max(cube_field2), 100 )
    
    if cube_weight_field is None:
        ret = stats.binned_statistic_2d(cube_field1, cube_field2, cube_field3, statistic=stat, bins=[f1_bin, f2_bin] )
        f3_aa = ret.statistic
    else:
        if cube_weight_field.ndim>2:
            cube_weight_field = cube_weight_field.flatten()
        ret1 = stats.binned_statistic_2d(cube_field1, cube_field2, cube_field3*cube_weight_field, statistic=stat, bins=[f1_bin, f2_bin] )
        ret2 = stats.binned_statistic_2d(cube_field1, cube_field2, cube_weight_field, statistic=stat, bins=[f1_bin, f2_bin] )
        f3_aa = ret1.statistic/ret2.statistic        

    f1_a = 1/2*(f1_bin[1:]+f1_bin[:-1])
    f2_a = 1/2*(f2_bin[1:]+f2_bin[:-1])
    
    f3_aa = f3_aa.transpose() # used for imshow
    return f1_a, f2_a, f3_aa

phase_diagram = general_2d_histogram


#########
#
#########

def cal_mdot(vr, mass, r, r_bin, geometry='sphere', in_out='outflow', v_cut=0, dL=0.1):
    '''
    mass (momentum or energy) inflow (or outflow) rate 
    vr in km/s
    '''
    N = len(r_bin)-1 # number of radial profile elements

    vr = vr.flatten()
    mass = mass.flatten()
    r = r.flatten()
    
    if dL is None:
        dL = r_bin[1:]-r_bin[:-1]
    elif hasattr(dL, '__len__') == False:
        dL = np.ones(N)*dL
    elif hasattr(dL, '__len__') == True:
        pass 
    
    if type(vr).__name__ != 'Quantity':
        vr = vr*u.km/u.s 
    if type(mass).__name__ != 'Quantity':
        mass = mass*u.g 
    if type(r).__name__ != 'Quantity':
        r = r*u.kpc
    if type(r_bin) != 'Quantity':
        r_bin *= u.kpc 
    if type(v_cut) != 'Quantity':
        v_cut *= u.km/u.s 
    if type(dL) != 'Quantity':
        dL *= u.kpc
        
    r_a = 1/2*(r_bin[1:]+r_bin[:-1])
    mdot_a = np.zeros( N )
    
    for i in range( N ):
        if in_out == 'outflow':
            idx = np.where( (r>r_bin[i]) & (r<r_bin[i+1]) & (vr>v_cut) )[0]
            mdot = np.nansum( vr[idx]*mass[idx] ) /dL[i] 

        elif in_out == 'inflow':
            idx = np.where( (r>r_bin[i]) & (r<r_bin[i+1]) & (vr<v_cut) )[0]
            mdot = -np.nansum( vr[idx]*mass[idx] ) /dL[i]   
            
        elif in_out=='two_cut':
            idx = np.where( (r>r_bin[i]) & (r<r_bin[i+1]) & (v_cut[0]<vr) & (vr<v_cut[1]) )[0]
            mdot = np.nansum( vr[idx]*mass[idx] ) /dL[i]     

        mdot = mdot.to(u.M_sun/u.yr).value   
        mdot_a[i] = mdot  

    if len(r_bin)>2:
        return r_a, mdot_a
    elif len(r_bin)==2:
        mdot = mdot_a[0]
        return mdot

def cal_sfr(star_age, star_mass, age_bin ):
    ''' 
    star_age: 1d array of stellar age
    star_mass: 1d array of star mass, unit should be solar mass
    age_bin: time bin used to calculate sfr, can be []

    Output:
    age_a, sfr_a if age_bin has more than elements 

    sfr if age_bin has 2 elements
    ''' 
    
    sfr_a = []
    for t1, t2 in zip(age_bin[:-1], age_bin[1:]):
        del_t = t2-t1
        idx = np.where( (t1<star_age) & (star_age<t2) )[0] # need modification
        M_star = np.nansum(star_mass[idx] )
        sfr = M_star/del_t
        sfr_a.append(sfr)
        
    if len(age_bin)>2:
        sfr_a = np.array(sfr_a)
        return sfr_a
    elif len(age_bin)==2:
        sfr = sfr_a[0]
        return sfr

def cal_sfr_img(star_age, star_mass, star_x, star_y, star_z, r_bin, age ):
    ''' 
    calculate spatially resolved image of sfr
    '''
    t1=0 
    t2=age
    del_t = t2-t1
    idx = np.where( (t1<star_age) & (star_age<t2) )[0] # need modification

    star_mass = star_mass[idx]
    star_x = star_x[idx]
    star_y = star_y[idx]
    star_z = star_z[idx]

    star_r = np.sqrt(star_x**2+star_y**2+star_z**2)

    r_a, Mstar_a = radial_profile(star_r, star_mass, None, r_bin, 'sum')
    area_a = np.pi*( r_bin[1:]**2 - r_bin[:-1]**2 )

    del_t *= 1e6 # in yr
    sfr_r = Mstar_a/area_a/del_t # in M_sun/yr
    sfr_r = np.array(sfr_r)
    return r_a, sfr_r

def construct_SFH(sim_suite_dir, sim_name, op_idx_a, halo_method, dt, ncpu_use, sim_code='RAMSES'):
    '''
    SFR averaged over dt (Myr)
    '''
    def cal_sfr_one_snap(op_idx):
        if sim_code=='RAMSES':
            # read_particle
            lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum, track_pos_halo_tracker, track_vel_halo_tracker \
                = anal_sim_util.get_sim_info(sim_suite_dir, sim_name, op_idx, halo_method=halo_method )
            _, _, _, _, _, _, redshiftnum_bef, _, _ \
                = anal_sim_util.get_sim_info(sim_suite_dir, sim_name, op_idx-1, halo_method=halo_method)

            size_kpc = 10
            size_cu = size_kpc/(lfac*cm2kpc)        

            p_imass, p_metal, pid, p_mass, p_x, p_y, p_z, p_rl, p_vx, p_vy, p_vz, p_tform, \
            star_imass, star_metal, star_id, star_mass, star_x, star_y, star_z, star_rl, \
            star_vx, star_vy, star_vz, star_age, dm_imass, dm_metal, dm_id, dm_mass, \
            dm_x, dm_y, dm_z, dm_rl, dm_vx, dm_vy, dm_vz = anal_sim_util.read_part_data(sim_suite_dir, sim_name, op_idx, size= size_cu, halo_method=halo_method)

        elif sim_code=='AREPO':
            raise Exception('Not implemented for AREPO yet!')

        # build SFH
        # SFR10
        time1 = cosmo.age(z=redshiftnum ).value
        time2 = cosmo.age(z=redshiftnum_bef ).value
        n_tp = int( ( (time1-time2)*1e3 )/dt ) + 1
        age_bin=[0]
        time_a4 = [] # sub time array between snapshots, in look back time, zero point at current time
        for i_tp in range(n_tp):
            age_bin.append( 10*(i_tp+1) )
            time_a4.insert(0, i_tp*10)

        time_a3 = cosmo.age(z=redshiftnum ).value - np.array( time_a4 )/1e3 # sub time array between snapshots, in forward time, zero point at beginning of universe
        
        sfr10 = cal_sfr(star_age, star_mass, age_bin=age_bin)/1e6 # in Msun/yr
        if hasattr(sfr10, '__len__'):
            sfr10 = np.flip(sfr10)
        else:
            pass
        return time_a3, sfr10

    time_a2=[]
    sfr10_tracker_a = []
    with Pool(ncpu_use) as pool:
        op = pool.map( cal_sfr_one_snap, op_idx_a )
        
    for i, op_idx in enumerate(op_idx_a):
        time_a_interval, sfr10 = op[i]
        time_a2.extend( time_a_interval )
        if hasattr(sfr10, '__len__'):
            sfr10_tracker_a.extend(sfr10)
        else:
            sfr10_tracker_a.append(sfr10)

    return time_a2, sfr10_tracker_a

def halo_anal(gas_r, gas_mass, part_r, part_mass, r_bin, z, verbose=False):
    # cube_r and part_r should be in unit of kpc
    # cube_mass and part_mass should be in unit of M_sun
    # return, density profile, virial quantities, rotation curve
    r_a = 1/2*(r_bin[1:]+r_bin[:-1])

    if gas_r is None and gas_mass is None:
        print('halo_anal: ignore gas component in the calculation')
        m = np.zeros_like( r_a )
    elif gas_r is not None and gas_mass is not None:
        if gas_r.ndim > 1 & gas_mass.ndim>1:
            gas_r = gas_r.flatten()
            gas_mass = gas_mass.flatten() 
        m, _, _ = stats.binned_statistic(gas_r, gas_mass, 'sum', r_bin)
    else:
        raise Exception('halo_anal: gas_r gas_mass format not correct!')

    m_part, _, _ = stats.binned_statistic(part_r, part_mass, 'sum', r_bin)
    
    # total mass within radius r
    M_r = np.zeros( len(r_bin) ) 
    M_r_g = np.zeros( len(r_bin) )
    M_r_p = np.zeros( len(r_bin) )
    for i in range( len(r_bin) ): 
        M_r[i] = np.nansum( m[:i] + m_part[:i] )
        M_r_g[i] = np.nansum( m[:i] )
        M_r_p[i] = np.nansum( m_part[:i] )

    
    # total density (dm + star + gas)
    dens_a = (m+m_part) /( 4/3*np.pi*(r_bin[1:]**3-r_bin[:-1]**3) )
    dens_a = (dens_a*u.M_sun/u.kpc**3).to(u.g/u.cm**3)
    dens_a = dens_a.value
    # virial quantities
    dens_thres = 200 * WMAP7.critical_density(z).value # in g/cm^3
    idx_a = np.where( dens_a > dens_thres)[0] 
    idx_vir = idx_a[-1]
    r_vir = r_a[ idx_vir ]    
    M_vir = M_r[ idx_vir ]
    V_vir = np.sqrt( c.G*M_vir*u.M_sun/(r_vir*u.kpc) ).to(u.km/u.s) 
    V_vir = V_vir.value

    # rotation curve
    v_c = np.sqrt(c.G*M_r[1:]*u.M_sun/r_bin[1:]/u.kpc).to(u.km/u.s)
    v_c_g = np.sqrt(c.G*M_r_g[1:]*u.M_sun/r_bin[1:]/u.kpc).to(u.km/u.s)
    v_c_p = np.sqrt(c.G*M_r_p[1:]*u.M_sun/r_bin[1:]/u.kpc).to(u.km/u.s)

    if verbose==True:
        plt.figure()
        plt.semilogy(r_bin, M_r, label='total mass')
        plt.semilogy(r_bin, M_r_g, label='gas mass')
        plt.semilogy(r_bin, M_r_p, label='star+dm mass')
        plt.xlabel('r [kpc]')
        plt.ylabel('mass [Msun]')
        plt.title('halo_anal: mass profile')
        plt.legend()

        plt.figure()
        plt.semilogy(r_a, dens_a)
        plt.semilogy(r_a, [dens_thres]*len(r_a) )
        plt.xlabel('r [kpc]')
        plt.ylabel(r'density [g/cm$^3$]')
        plt.title('halo_anal: density profile')
        print('Virial radius %.2f kpc'%r_vir, 'Virial mass %.2e M_sun'%M_vir, 'Virial velocity %.2f km/s'%V_vir)

        plt.figure()
        plt.plot(r_a, v_c, label='total')
        plt.plot(r_a, v_c_g, label='gas')
        plt.plot(r_a, v_c_p, label='star+dm')
        
    return r_a, dens_a, r_vir, M_vir, V_vir, v_c_g, v_c_p, v_c



    

