
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
import h5py

from num_util.const_util import *
from num_util.path_info import *

import num_util 

import plotbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from plotbin.display_bins import *


#############
# content 
# usual util
# radial profile analysis
# image analysis
# spectra analysis
# measurement related util
#############

def clean_data(x, set_value=0):
    '''
    remove nan, +- inf value in array x
    '''
    idx = np.where(x!=x)[0]
    N_nan = len(idx)
    if N_nan>0:
        print(N_nan, 'NaN in array, at index', idx)
        x[idx]=set_value
    idx = np.where(x==np.inf)[0]
    N_inf = len(idx)
    if N_inf>0:
        print(N_inf, '+inf in array, at index', idx)
        x[idx]=set_value
    idx = np.where(x==-np.inf)[0]
    N_minf = len(idx)
    if N_minf>0:
        print(N_minf, '-inf in array, at index', idx)
        x[idx]=set_value
    return x

##############
# usual util #
##############

# doppler effect
# v positive if gas moving away from us
def v2lam(v_a, lam0, value=True):
    '''
    v in km/s
    lam_a is in the same unit of lam0
    '''
    if type(v_a) in [float, np.float64, np.ndarray]:
        v_a *= u.km/u.s 
    if type(lam0) in [float, np.float64]:
        lam0 *= u.Angstrom    
    vabs_max = np.max( np.abs(v_a/c.c) )
    if vabs_max<=0.01:
        lam_a = lam0*(1 + v_a/c.c )
    elif 0.01<vabs_max and vabs_max<1:
        lam_a = lam0*np.sqrt( (1+v_a/c.c)/(1-v_a/c.c) )
    if value: lam_a = lam_a.value 
    return lam_a
v_to_lam = v2lam

def lam2v(lam_a, lam0, value=True):
    v_a = c.c*( lam_a/lam0 - 1 )
    vabs_max = np.max( np.abs(v_a/c.c) )
    if vabs_max>0.01:
        v_a = (lam_a**2-lam0**2)/( lam_a**2+lam0**2 ) * c.c
    v_a = v_a.to(u.km/u.s)
    if value: v_a = v_a.value
    return v_a
lam_to_v = lam2v

def v2nu(v_a, nu_0):
    if type(v_a)==np.ndarray:
        v_a = v_a*u.km/u.s 
    if type(nu_0)==float:
        nu_0 *= u.Hz
    vabs_max = np.max( np.abs(v_a/c.c) )
    if vabs_max<=0.01:
        nu_a = nu_0*(1 - v_a/c.c)
    elif 0.01<vabs_max and vabs_max<1:
        nu_a = nu_0*np.sqrt( (1-v_a/c.c) / (1+v_a/c.c) ) 
    nu_a = nu_a.to(u.Hz)
    return nu_a
v_to_nu=v2nu

def nu2v(nu_a, nu_0):
    v_a = c.c*(1-nu_a/nu_0)
    vabs_max = np.max( np.abs(v_a/c.c) )
    if vabs_max>0.01:
        v_a = ( nu_0**2-nu_a**2 ) / ( nu_0**2+nu_a**2 ) *c.c 
    v_a = v_a.to(u.km/u.s)
    return v_a   
nu_to_v=nu2v 
     
def lam2nu(lam):
    assert lam.unit.is_equivalent(u.Angstrom)
    nu = c.c/lam
    return nu 
lam_to_nu = lam2nu

def nu2lam(nu):
    assert nu.unit.is_equivalent(u.Hz)
    lam = c.c/nu
    return lam
nu_to_lam=nu2lam


def cal_vth(m, T):
    vth = np.sqrt( 2*c.k_B*T/m )
    vth = vth.to(u.cm/u.s)
    return vth 

def cal_Dopwidth(x0, m, T):
    '''
    Input
        - x0: central wavelength or frequency of the line.
        - m: mass of particle
        - T: temperature
    Output:
        - sigma: dispersion
    '''
    vth = cal_vth(m, T)
    sigma = x0 * vth /c.c
    sigma.to(x0.unit)
    return sigma 

def cal_ion_number_density(dens, abundance_a, gas_comp = 'H_He'):
    '''
    gas composition of H and He. 
    Input:
      dens: mass density
      abundance_a: array of abundance of various species
      gas_comp: gas composition
    Output:
        - number density of various types and species 
    '''
    if gas_comp=='H_He':
        xHII, xHeII, xHeIII = abundance_a        
        ndens = dens/c.m_p * ( f_H*(1+xHII) + f_He/4*(1+xHeII+2*xHeIII) ) # including electron 
        nH = dens*f_H/c.m_p
        nHI = nH*(1-xHII)
        nHII = nH*xHII
        nHe = 0.25*nH*f_He/f_H
        nHeII = nHe*xHeII
        nHeIII = nHe*xHeIII
        nHeI = nHe - nHeII - nHeIII
        ne = nHII + nHeII + nHeIII*2
        
        ndens, nH, nHI, nHII, nHe, nHeI, nHeII, nHeIII, ne = ndens.to(u.cm**-3), nH.to(u.cm**-3), nHI.to(u.cm**-3), nHII.to(u.cm**-3), nHe.to(u.cm**-3), nHeI.to(u.cm**-3), nHeII.to(u.cm**-3), nHeIII.to(u.cm**-3), ne.to(u.cm**-3)
        return ndens, nH, nHI, nHII, nHe, nHeI, nHeII, nHeIII, ne

def cal_TK(p_th, ndens):
    '''
    Calculate temperature
    inputs:
        - p_th: thermal pressure
        - ndens: number density
    Outputs:
        - TK: temperature
    '''
    TK = p_th/(ndens*c.k_B )
    TK = TK.to(u.K) 
    return TK

def cal_xion(NH0, L1500):
    '''
    eq (1) in Saxena+2024
    eq (5) in Simmonds+2023
    ionising photon production efficiency
    Both NH0 and L1500 is intrinsic.
    '''
    assert NH0.unit.is_equivalent(u.Hz) and L1500.unit.is_equivalent(u.erg/u.s/u.Hz)
    xion = (NH0/L1500).to(u.erg**-1*u.Hz) 
    return xion 
    
    

def ang_to_d(theta, z, cosmo):
    '''
    convert angular separation to distance:
    Inputs:
        theta: angular distance 
        z: redshift
        cosmo: cosmology object 
    Outputs:
        d: distance
    '''
    DA = cosmo.angular_diameter_distance(z)
    theta = theta.to(u.rad).value
    d = DA*theta
    d = d.to(u.kpc)
    return d

def d_to_ang():
    '''
    convert the 
    '''







def cal_mdot(vr, mass, r, r_bin, geometry='sphere', in_out='outflow', v_cut=0):
    '''
    calculate mass (momentum or energy) inflow (or outflow) rate 
    Inputs:
    Outputs:
        - mdot_a if r_bin has more than 2 elements; mdot if r_bin has only 2 elements. 
    '''
    N = len(r_bin)-1 # number of radial profile elements

    vr = vr.flatten()
    mass = mass.flatten()
    r = r.flatten()
    
    if type(vr).__name__ != 'Quantity':
        vr = vr*u.km/u.s 
    if type(mass).__name__ != 'Quantity':
        mass = mass*u.g 
    if type(r).__name__ != 'Quantity':
        r = r*u.kpc
    if type(r_bin).__name__ != 'Quantity':
        r_bin *= u.kpc 
    if type(v_cut).__name__ != 'Quantity':
        v_cut *= u.km/u.s 
        
    dL = r_bin[1:]-r_bin[:-1]
    
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



def cal_sfr(star_age, star_mass, age_bin, op_unit=False ):
    ''' 
    Inputs:
        - star_age: 1d array of stellar age
        - star_mass: 1d array of star mass, unit should be solar mass
        - age_bin: stellar age bin used to calculate sfr.
    Output:
        -if age_bin has only 2 elements:      sfr (float) 
        -if age_bin has more than 2 elements: age_a, sfr_a (array) 
    ''' 
    
    sfr_a = []
    for t1, t2 in zip(age_bin[:-1], age_bin[1:]):
        del_t = t2-t1
        idx = np.where( (t1<star_age) & (star_age<t2) )[0] # need modification
        M_star = np.nansum(star_mass[idx] )
        sfr = M_star/del_t
        if type(sfr)==u.quantity.Quantity:
            sfr = sfr.to(u.M_sun/u.yr).value
        sfr_a.append(sfr)
        
    if len(age_bin)>2:
        sfr_a = np.array(sfr_a)
        if op_unit:
            sfr_a *= u.M_sun/u.yr
        return sfr_a
    elif len(age_bin)==2:
        sfr = sfr_a[0]
        if op_unit:
            sfr *= u.M_sun/u.yr
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

def cal_sfr_one_snap(sim_suite_dir, sim_name, op_idx, halo_method, dt, sim_code):
    if sim_code=='RAMSES':
        # read_particle
        lfac, dfac, tfac, aexp, ctime, redshift, redshiftnum, track_pos_halo_tracker, track_vel_halo_tracker \
            = num_util.get_sim_info(sim_suite_dir, sim_name, op_idx, halo_method=halo_method )
        _, _, _, _, _, _, redshiftnum_bef, _, _ \
            = num_util.get_sim_info(sim_suite_dir, sim_name, op_idx-1, halo_method=halo_method)

        size_kpc = 10
        size_cu = size_kpc/(lfac*cm2kpc)        

        p_imass, p_metal, pid, p_mass, p_x, p_y, p_z, p_rl, p_vx, p_vy, p_vz, p_tform, \
        star_imass, star_metal, star_id, star_mass, star_x, star_y, star_z, star_rl, \
        star_vx, star_vy, star_vz, star_age, dm_imass, dm_metal, dm_id, dm_mass, \
        dm_x, dm_y, dm_z, dm_rl, dm_vx, dm_vy, dm_vz = num_util.read_part_data(sim_suite_dir, sim_name, op_idx, size= size_cu, halo_method=halo_method)

    elif sim_code=='AREPO':
        raise Exception('Not implemented for AREPO yet!')

    # build SFH
    # SFR10
    time1 = cosmo.age(z=redshiftnum ).value # in Gyr
    time2 = cosmo.age(z=redshiftnum_bef ).value # in Gyr
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

    
def construct_SFH(sim_suite_dir, sim_name, op_idx_a, halo_method, dt, ncpu_use, sim_code='RAMSES'):
    '''
    in-situ SFR averaged over dt (Myr)
    
    Input:
    '''
    time_a=[]
    sfr10_tracker_a = []
    
    ip_var = []
    for op_idx in op_idx_a:
        ip_var.append( (sim_suite_dir, sim_name, op_idx, halo_method, dt, sim_code) )
    
    with Pool(ncpu_use) as pool:
        op = pool.starmap( cal_sfr_one_snap, ip_var )
        
    for i, op_idx in enumerate(op_idx_a):
        time_a_interval, sfr10 = op[i]
        time_a.extend( time_a_interval )
        if hasattr(sfr10, '__len__'):
            sfr10_tracker_a.extend(sfr10)
        else:
            sfr10_tracker_a.append(sfr10)

    time_a = np.array(time_a)
    sfr10_tracker_a = np.array(sfr10_tracker_a)

    return time_a, sfr10_tracker_a

def SFH_from_last_snap(massstar, agestar, dt, tlast):
    '''
    Construct SFH history from the last snapshot
    suppose the system in the last snapshot is S
    SFH of all progenitors of S.
    '''
    age_max = np.max(agestar)
    Nt = np.ceil( (age_max/dt).decompose().value  )
    Nt = int(Nt)
    age_max2 = Nt*dt
    age_bin = np.linspace(0, age_max2, Nt+1)
    
    SFR_a = cal_sfr(agestar, massstar, age_bin, op_unit=True) 
    time_a = tlast-age_bin[:-1]
    return time_a, SFR_a

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
        
    return r_a, dens_a, r_vir, M_vir, V_vir, v_c_g, v_c_p, v_c

####################
# Bayesian         #
####################

def cal_log_likelihood(f, f_fit, sigma):
    '''
    calculate the log of the likelihood function
    Input:
      f: array
      f_fit: array
      sigma: can be a number or array
    Output:
      log_ll
    '''
    if sigma is None:
        sigma=1
    kaisq = np.nansum( (f-f_fit)**2/sigma**2 )
    kaisq_by_N = kaisq/len(f)
    log_ll = -0.5*kaisq
    return kaisq_by_N,log_ll

def cal_AIC(k, log_ll ):
    '''
    Input:
      k:      number of variables
      log_ll: log of maximum value of likelihood function
    Output:
      AIC
    '''
    AIC = 2*k - 2*log_ll
    return AIC

def cal_Akaike_weight(AIC_a):
    pass

###########################
# radial profile analysis # 
###########################

# notes on curve_fit, value of y array should be similar to 1 (order of magnitude). If y array is overall to big or small, either rescale or do log 
def fit_Sersic(r_a, I_r, log=True, verbose=False):
    if log==True:
        r_a = r_a[I_r>0]
        I_r = I_r[I_r>0]
        print(3*np.max(I_r), np.max(r_a ), 10)
        popt, pcov = curve_fit(num_util.log_Sersic, r_a, np.log10(I_r), bounds=(0, [10*np.max(I_r), np.max(r_a ), 10])  ) 
    else:
        popt, pcov = curve_fit(num_util.Sersic, r_a, I_r)
    A, r_eff, n = popt
    
    if verbose:
        plt.figure()
        print('fitted parameters', A, r_eff, n)
        plt.semilogy(r_a, I_r )
        plt.semilogy(r_a, num_util.Sersic(r_a, A, r_eff, n), label='Sersic' )
        plt.legend()
    return A, r_eff, n

def fit_exp(r_a, I_r, sigma_r=0):
    popt, pcov = curve_fit(num_util.exp, r_a, I_r)
    A, rs = popt
    return A, rs

def fit_dbl_exp(r_a, I_r, sigma_r=0, log=True, verbose=False):
    if log==True:
        r_a = r_a[I_r>0]
        I_r = I_r[I_r>0]
        popt, pcov = curve_fit(num_util.log_double_exp, r_a, np.log10(I_r), bounds=([np.log10( np.max(I_r)/100 ), 0,0,0], [ np.log10( 10*np.max(I_r) ), np.max(r_a), 3*np.max(r_a), np.max(r_a)  ])  )
    elif log==False:
        popt, pcov = curve_fit(num_util.double_exp, r_a, I_r)
    logA1, r1, r2, b = popt
    A1 = 10**logA1
    A2 = A1*np.exp( b/r2 - b/r1 )
    
    if verbose:
        print('fitted parameter', A1, A2, r1, r2, b)
        plt.figure(100)
        if log:
            plt.semilogy(r_a, I_r)
            plt.semilogy(r_a, num_util.double_exp(r_a, A1, r1, r2, b), label='double exponential' )
        else:
            plt.plot(r_a, I_r)
            plt.plot(r_a, num_util.double_exp(r_a, A1, r1, r2, b), label='double power law' )
        plt.legend()
        plt.clf()
    return A1, A2, r1, r2, b

####################
# spectra analysis #
####################
def fit_gaussian(x_dat, y_dat, bounds=None, verbose=False):
    x_max = np.max( np.abs(x_dat) )
    y_max = np.max( np.abs(y_dat) )
    
    x_dat_n = x_dat/x_max 
    y_dat_n = y_dat/y_max
    
    Delta_x_dat_n = np.max(x_dat_n) - np.min(x_dat_n)
    
    if bounds is None:
        bounds=([0.5, np.min(x_dat_n), 0],
                [1.5, np.max(x_dat_n), Delta_x_dat_n])
    popt, pcov = curve_fit(num_util.gaussian, x_dat_n, y_dat_n, bounds=bounds )
    A, x0, sigma = popt
    if verbose: print('fit_gaussian: A, x0, sigma', A, x0, sigma)
    
    A *= y_max
    x0 *= x_max
    sigma *= x_max
    return A, x0, sigma



def fit_double_gaussian(x_dat, y_dat, x_cen=0, bounds=None):
    '''
    Fit double gaussian
    '''
    x_max = np.max( np.abs(x_dat) )
    y_max = np.max( np.abs(y_dat) )
    
    x_dat = x_dat/x_max 
    y_dat = y_dat/y_max
    x_cen = x_cen/x_max 
    
    idx = np.where( x_dat<x_cen )[0]
    x_mean_1 = np.trapz( y_dat[idx]*x_dat[idx], x_dat[idx] ) / np.trapz( y_dat[idx], x_dat[idx] )
    
    idx = np.where( x_dat>x_cen )[0]
    x_mean_2 = np.trapz( y_dat[idx]*x_dat[idx], x_dat[idx] ) / np.trapz( y_dat[idx], x_dat[idx] )
    if bounds is None:
        x0_1_n_min=x_mean_1-0.2
        x0_1_n_max=min(x_cen, x_mean_1+0.2)
        x0_2_n_min=max(x_cen, x_mean_2-0.2)
        x0_2_n_max=x_mean_2+0.2

        bounds= ([0.0, x0_1_n_min, 0.00,   0.0, x0_2_n_min, 0.00],
                 [1.2, x0_1_n_max, 0.20,   1.2, x0_2_n_max, 0.20])
    popt, pcov = curve_fit(num_util.double_gaussian, x_dat, y_dat, bounds=bounds )
    A1, x0_1, sigma1, A2, x0_2, sigma2 = popt
    
    A1 *= y_max
    x0_1 *= x_max
    sigma1 *= x_max
    
    A2 *= y_max
    x0_2 *= x_max
    sigma2 *= x_max
    return A1, x0_1, sigma1, A2, x0_2, sigma2


def asymmetric_gaussian(lam_a, A, lam_0_asym, a_asym, d, modify='a' ):
    '''
    Modified eq (2) in Shibuya+14b
    '''

    sig_asym = a_asym*(lam_a - lam_0_asym)+d
    
    sig_min = 1e-3*d
    sig_asym[sig_asym<sig_min] = sig_min
        
    if modify=='a':
        sig_max = 1e1*d
        sig_asym[sig_asym>sig_max] = sig_max
        f_damp = 1
    elif modify=='b':
        # damp the long tail part 
        def cal_f_damp(lam, lam_0, a, d):
            if a>0:
                if lam<=lam_0:
                    f = 1
                else:
                    f = np.exp( -np.abs(lam-lam_0)/(10*d) )
            elif a==0:
                f = 1
            else:
                # a<0
                if lam>=lam_0:
                    f=1
                else:
                    f = np.exp( -np.abs(lam-lam_0)/(10*d) )
            return f 
        cal_f_damp = np.vectorize(cal_f_damp)
        
        f_damp = cal_f_damp(lam_a, lam_0_asym, a_asym, d) 
    else:
        raise Exception('No this modification')
    
    f = A*np.exp( -(lam_a-lam_0_asym)**2 / (2*sig_asym**2) ) 
    f = f*f_damp
    return f

def double_asymmetric_gaussian(lam_a, A1, lam_1_asym, a_1_asym, d1, A2, lam_2_asym, a_2_asym, d2):
    I1 = asymmetric_gaussian(lam_a, A1, lam_1_asym, a_1_asym, d1)
    I2 = asymmetric_gaussian(lam_a, A2, lam_2_asym, a_2_asym, d2)
    I=I1+I2  
    return I

def triple_asymmetric_gaussian(lam_a, A1, lam_1_asym, a_1_asym, d1, A2, lam_2_asym, a_2_asym, d2, A3, lam_3_asym, a_3_asym, d3):
    I1 = asymmetric_gaussian(lam_a, A1, lam_1_asym, a_1_asym, d1)
    I2 = asymmetric_gaussian(lam_a, A2, lam_2_asym, a_2_asym, d2)
    I3 = asymmetric_gaussian(lam_a, A3, lam_3_asym, a_3_asym, d3)
    I=I1+I2+I3  
    return I


def fit_single_asymmetric_gaussian(lam_a, I_a, sigma=None, verbose=False, log=False):
    if log==False:
        nor = np.max(I_a)
        I_a_nor = I_a/nor
        sigma = sigma/nor

        popt, pcov, infodict, mesg, ier = curve_fit(asymmetric_gaussian, lam_a, I_a_nor, sigma=sigma, \
            bounds=( [ 0.05, np.percentile(lam_a, 25, ), -0.5, 0.03], \
                     [ 1.2, np.percentile(lam_a, 75, ),  +0.5, 1.5] ),
            full_output=True )
        A, lam_asym, a_asym, d = popt

        I_fit = asymmetric_gaussian(lam_a, A, lam_asym, a_asym, d)
        kaisq_by_N, log_ll = cal_log_likelihood(I_a_nor, I_fit, sigma)
        AIC = cal_AIC(4, log_ll)

        A *= nor
        I_fit *=nor

    if verbose:
        print('AIC:', AIC)
        print('fitted parameters A %.2e, lam_asym %.2f, a_asym %.2f, d %.2f'%(A, lam_asym, a_asym, d) )
    return A, lam_asym, a_asym, d, I_fit, AIC


def fit_double_asymmetric_gaussian(lam_a, I_a, sigma=None, verbose=False, log=False):
    '''
    Fit double 
    '''
    if log==False:
        nor = np.max(I_a)
        I_a_nor = I_a/nor
        sigma = sigma/nor

        popt, pcov, infodict, mesg, ier = \
            curve_fit(double_asymmetric_gaussian, lam_a, I_a_nor, sigma=sigma, \
                        bounds=( [ 0.05, np.percentile(lam_a, 25, ),  -0.5, 0.03, \
                                   0.05, np.percentile(lam_a, 40),    0,   0.03], \
                                 [ 1.2,  np.percentile(lam_a, 60, ),  0,   1.5,   \
                                   1.2,  np.percentile(lam_a, 75),    0.5, 1.5] ), \
                        full_output=True )
        
        A1, lam_1_asym, a_1_asym, d1, A2, lam_2_asym, a_2_asym, d2 = popt

        if lam_1_asym>lam_2_asym:
            A1, lam_1_asym, a_1_asym, d1, A2, lam_2_asym, a_2_asym, d2 = A2, lam_2_asym, a_2_asym, d2, A1, lam_1_asym, a_1_asym, d1

        I_fit = double_asymmetric_gaussian(lam_a, A1, lam_1_asym, a_1_asym, d1, A2, lam_2_asym, a_2_asym, d2)
        kaisq_by_N, log_ll = cal_log_likelihood(I_a_nor, I_fit, sigma)
        AIC = cal_AIC(8, log_ll)

        A1 *= nor
        A2 *= nor
        I_fit *=nor

    if verbose:
        print('AIC:', AIC)
        print('fitted parameters A1 %.2e, lam_1_asym %.2f, a_1_asym %.2f, d1 %.2f'%(A1, lam_1_asym, a_1_asym, d1) )
        print('fitted parameters A2 %.2e, lam_2_asym %.2f, a_2_asym %.2f, d2 %.2f'%(A2, lam_2_asym, a_2_asym, d2) )
    return A1, lam_1_asym, a_1_asym, d1, A2, lam_2_asym, a_2_asym, d2, I_fit, AIC


def fit_triple_asymmetric_gaussian(lam_a, I_a, sigma=None, verbose=False, log=False):
    if log==False:
        nor = np.max(I_a)
        I_a_nor = I_a/nor
        sigma = sigma/nor

        popt, pcov, infodict, mesg, ier = curve_fit(triple_asymmetric_gaussian, lam_a, I_a_nor, sigma=sigma, \
            bounds=( [0.05, np.percentile(lam_a, 20), -0.5, 0.03,    \
                      0.05, np.percentile(lam_a, 30), -0.5, 0.03,    \
                      0.05, np.percentile(lam_a, 40), 0,    0.03  ], \
                     [1.2, np.percentile(lam_a, 60),  0,    1.5,    \
                      1.2, np.percentile(lam_a, 70),  0.5,  1.5,    \
                      1.2, np.percentile(lam_a, 80),  0.5,  1.5  ] ),
            full_output=True )
        A1, lam_1_asym, a_1_asym, d1, A2, lam_2_asym, a_2_asym, d2, A3, lam_3_asym, a_3_asym, d3 = popt

        pk_list = [(A1, lam_1_asym, a_1_asym, d1), (A2, lam_2_asym, a_2_asym, d2), (A3, lam_3_asym, a_3_asym, d3)]
        pk1, pk2, pk3 = sorted(pk_list, key=lambda pk: pk[1] )
        A1, lam_1_asym, a_1_asym, d1 = pk1 
        A2, lam_2_asym, a_2_asym, d2 = pk2
        A3, lam_3_asym, a_3_asym, d3 = pk3       

        I_fit = triple_asymmetric_gaussian(lam_a, A1, lam_1_asym, a_1_asym, d1, A2, lam_2_asym, a_2_asym, d2, A3, lam_3_asym, a_3_asym, d3)
        kaisq_by_N, log_ll = cal_log_likelihood(I_a_nor, I_fit, sigma)
        AIC = cal_AIC(12, log_ll)

        A1 *= nor  
        A2 *= nor
        A3 *= nor
        I_fit *=nor

    if verbose:        
        print('AIC:', AIC)
        print('fitted parameters A1 %.2e, lam_1_asym %.2f, a_1_asym %.2f, d1 %.2f'%(A1, lam_1_asym, a_1_asym, d1) )
        print('fitted parameters A2 %.2e, lam_2_asym %.2f, a_2_asym %.2f, d2 %.2f'%(A2, lam_2_asym, a_2_asym, d2) )
        print('fitted parameters A3 %.2e, lam_3_asym %.2f, a_3_asym %.2f, d3 %.2f'%(A3, lam_3_asym, a_3_asym, d3) )
    return A1, lam_1_asym, a_1_asym, d1, A2, lam_2_asym, a_2_asym, d2, A3, lam_3_asym, a_3_asym, d3, I_fit, AIC




def cal_vsep(npk, *arg, **kwargs):
    if 'verbose' in kwargs.keys():
        verbose = kwargs['verbose']
    else:
        verbose=False
        
    if npk==1:
        single_peak=True
        lam, vcen = arg
        v_pk = lam_to_v(lam, lam_lya)

    elif npk==2:
        A1, lam1, a_1_asym, d1, A2, lam2, a_2_asym, d2, vcen, lam_a = arg
        v1 = lam_to_v(lam1, lam_lya)
        v2 = lam_to_v(lam2, lam_lya)
        
        f1 = asymmetric_gaussian(lam_a, A1, lam1, a_1_asym, d1)
        f2 = asymmetric_gaussian(lam_a, A2, lam2, a_2_asym, d2)
        
        S1 = np.trapz(f1, lam_a)
        S2 = np.trapz(f2, lam_a)
        
        if v1<vcen and v2>vcen:
            single_peak=False
            vblue=v1
            vred=v2
            vsep = vred-vblue
        elif v2<=vcen or v1>=vcen:
            if verbose: print('astro_util.cal_vsep:double peak on one side')
            single_peak=True
            v_pk = (v1*S1 + v2*S2)/(S1+S2)
        else:
            single_peak=False 
            raise Exception('something is wrong')

    elif npk==3:
        A1, lam1, a_1_asym, d1, A2, lam2, a_2_asym, d2, A3, lam3, a_3_asym, d3, vcen, lam_a = arg
        v1 = lam_to_v(lam1, lam_lya)
        v2 = lam_to_v(lam2, lam_lya)
        v3 = lam_to_v(lam3, lam_lya)
        
        f1 = asymmetric_gaussian(lam_a, A1, lam1, a_1_asym, d1)
        f2 = asymmetric_gaussian(lam_a, A2, lam2, a_2_asym, d2)
        f3 = asymmetric_gaussian(lam_a, A3, lam3, a_3_asym, d3)
        
        S1 = np.trapz(f1, lam_a)
        S2 = np.trapz(f2, lam_a)
        S3 = np.trapz(f3, lam_a)
        
        if v1<vcen and v2<vcen and v3>vcen:
            single_peak=False
            vblue = (v1*S1+v2*S2)/(S1+S2)
            vred  = v3
            vsep = vred - vblue 
        elif v1<vcen and v2>vcen and v3>vcen:
            single_peak=False
            vred = (v2*S2+v3*S3)/(S2+S3)
            vblue=v1
            vsep = vred - vblue 
        elif v3<=vcen or v1>=vcen:
            if verbose: print('astro_util.cal_vsep:triple peak on one side')
            single_peak=True
            v_pk = (v1*S1 + v2*S2 + v3*S3)/(S1+S2+S3)
        else:
            single_peak=False
            raise Exception('something is wrong')
        
    else:
        raise Exception('npk should be 1,2 or 3, but npk is', npk)
            
    if single_peak==True:
        if v_pk<vcen:
            vblue=v_pk
            vred=None 
        elif v_pk>=vcen:
            vblue=None
            vred=v_pk       
        vsep = 2*np.abs( v_pk-vcen )    
    else:
        pass
     
    return vblue, vred, vsep 

def cal_r2b_fcen(lam_a, spectrum, vcen):
    v = lam_to_v(lam_a, lam_lya)
    ridx = v>vcen
    bidx = v<vcen
    cenidx = np.where( np.abs(v-vcen)<40 )
    r2b = np.trapz( spectrum[ridx], v[ridx] ) / np.trapz( spectrum[bidx], v[bidx] )
    fcen = np.trapz( spectrum[cenidx], v[cenidx] ) / np.trapz( spectrum, v )
    return r2b, fcen 

def cal_Af(lam_a, spectrum, vcen, vred):
    '''
    Calculate the asymmetry parameter of the red/blue peak, according to the definition eq (33) in Kakiichi & Gronke 2021.
    '''
    if vred is None:
        Af = np.nan
    else:
        lam_cen = v_to_lam(vcen, lam_lya)
        lam_red = v_to_lam(vred, lam_lya)
        rr_i = np.where( lam_a>lam_red ) # red right 
        rl_i = np.where( (lam_cen<lam_a) & (lam_a<lam_red) ) # red left
        Af = np.trapz( spectrum[rr_i], lam_a[rr_i] ) / np.trapz( spectrum[rl_i], lam_a[rl_i] )    
    return Af

def cal_lya_line_parameter(lam_a, spectrum, sigma=None, verbose=True, vcen=None):
    '''
    calculate Lya line parameters
    '''

    def fit_Lya_profile(lam_a, I_a, sigma=None, verbose=True, AIC_thres=np.inf ):
        '''
        Fit Lya profile with single, double or triple peaked profile and select the best case
        '''
        try:
            sg_A, sg_lam_asym, sg_a_asym, sg_d, sg_I_fit, sg_AIC = fit_single_asymmetric_gaussian(lam_a, I_a, sigma=sigma, verbose=verbose, log=False)
        except Exception as e:
            print('astro_util.fit_Lya_profile, error:',e)
            sg_AIC = np.inf
        try:    
            db_A1, db_lam_1_asym, db_a_1_asym, db_d1, db_A2, db_lam_2_asym, db_a_2_asym, db_d2, db_I_fit, db_AIC = fit_double_asymmetric_gaussian(lam_a, I_a, sigma=sigma, verbose=verbose, log=False)
        except Exception as e:
            print('astro_util.fit_Lya_profile, error:',e)
            db_AIC= np.inf
        try:
            tp_A1, tp_lam_1_asym, tp_a_1_asym, tp_d1, tp_A2, tp_lam_2_asym, tp_a_2_asym, tp_d2, tp_A3, tp_lam_3_asym, tp_a_3_asym, tp_d3, tp_I_fit, tp_AIC = fit_triple_asymmetric_gaussian(lam_a, I_a, sigma=sigma, verbose=verbose, log=False)
        except Exception as e:
            print('astro_util.fit_Lya_profile, error:',e)
            tp_AIC = np.inf

        fit_list = [(1, sg_AIC), (2, db_AIC), (3, tp_AIC)]
        npk, AIC_min = sorted(fit_list, key=lambda f: f[1] )[0] # find the best fitting models, i.e. peak number
        if AIC_min>AIC_thres:
            print('astro_util.fit_Lya_profile: AIC_min %.1f is larger than AIC_thres %.1f so no fitting results'%(AIC_min, AIC_thres) )
            npk=0
            par=None
            I_fit = np.zeros_like(I_a)

        if npk==1:
            par=sg_A, sg_lam_asym, sg_a_asym, sg_d
            I_fit = sg_I_fit
        elif npk==2:
            par = db_A1, db_lam_1_asym, db_a_1_asym, db_d1, db_A2, db_lam_2_asym, db_a_2_asym, db_d2
            I_fit = db_I_fit 
        elif npk==3:
            par = tp_A1, tp_lam_1_asym, tp_a_1_asym, tp_d1, tp_A2, tp_lam_2_asym, tp_a_2_asym, tp_d2, tp_A3, tp_lam_3_asym, tp_a_3_asym, tp_d3
            I_fit = tp_I_fit
        return npk, par, I_fit

    npk, par, I_fit = fit_Lya_profile(lam_a, spectrum, sigma=sigma, verbose=verbose)
    if npk==0:
        vsep, peak_ratio, r2b, fcen, sigma_red, Af = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    elif npk==1:
        A, lam_asym, a_asym, d = par 
        vblue, vred, vsep = cal_vsep(npk, lam_asym, vcen)
        peak_ratio = np.nan
        r2b, fcen = cal_r2b_fcen(lam_a, spectrum, vcen)
        sigma_red = 3e5*d/lam_lya 
        
    elif npk==2:
        A1, lam_1_asym, a_1_asym, d1, A2, lam_2_asym, a_2_asym, d2 = par
        vblue, vred, vsep = cal_vsep(npk, A1, lam_1_asym, a_1_asym, d1, A2, lam_2_asym, a_2_asym, d2, vcen, lam_a)
        peak_ratio = A2/A1
        r2b, fcen = cal_r2b_fcen(lam_a, spectrum, vcen)
        sigma_red = 3e5*d2/lam_lya 

    elif npk==3:
        A1, lam_1_asym, a_1_asym, d1, A2, lam_2_asym, a_2_asym, d2, A3, lam_3_asym, a_3_asym, d3 = par
        vblue, vred, vsep  = cal_vsep(npk, A1, lam_1_asym, a_1_asym, d1, A2, lam_2_asym, a_2_asym, d2, A3, lam_3_asym, a_3_asym, d3, vcen, lam_a)
        peak_ratio=A3/A1 
        r2b, fcen = cal_r2b_fcen(lam_a, spectrum, vcen)
        sigma_red = 3e5*d3/lam_lya 
    
    Af = cal_Af(lam_a, spectrum, vcen, vred)
    return npk, vblue, vred, vsep, peak_ratio, r2b, fcen, sigma_red, Af, I_fit



def cal_lya_line_parameter_single(lam_a, spectrum, sigma=None, verbose=True, vcen=None):
    '''
    calculate Lya line parameters
    '''

    def fit_Lya_profile_single(lam_a, I_a, sigma, verbose):
        '''
        Fit Lya profile with single, double or triple peaked profile and select the best case
        '''
        try:
            sg_A, sg_lam_asym, sg_a_asym, sg_d, sg_I_fit, sg_AIC = fit_single_asymmetric_gaussian(lam_a, I_a, sigma=sigma, verbose=verbose, log=False)
        except Exception as e:
            print('astro_util.fit_Lya_profile, error:',e)
            sg_AIC = np.inf

        npk=1
        par=sg_A, sg_lam_asym, sg_a_asym, sg_d
        I_fit = sg_I_fit
        return npk, par, I_fit

    npk, par, I_fit = fit_Lya_profile_single(lam_a, spectrum, sigma, verbose)

    A, lam_asym, a_asym, d = par 
    vblue, vred, vsep = cal_vsep(npk, lam_asym, vcen)
    peak_ratio = np.nan
    r2b, fcen = cal_r2b_fcen(lam_a, spectrum, vcen)
    sigma_red = 3e5*d/lam_lya 
        
    Af = cal_Af(lam_a, spectrum, vcen, vred)
    return npk, vblue, vred, vsep, peak_ratio, r2b, fcen, sigma_red, Af, I_fit




def peak_analysis(spectrum_a, v, vsys_a=None, verbose=False, method='lya_fitting_complex', noise=None, op='std'):
    '''
    function to calcuate line parameters for Lya double peak spectra
    Input:
    '''
    
    nspec = np.shape(spectrum_a)[0]
    assert len(vsys_a)==nspec

    assert v.ndim in [1,2], 'Invalid dimension for v'

    if v.ndim == 1:
        nv = len(v)
    elif v.ndim == 2:
        nv = v.shape[1]
    lam = v_to_lam(v, lam_lya )
           
    if noise is None:
        sigma_a = 0.05*np.max( spectrum_a, axis=1 )
        sigma_a = sigma_a.reshape( (nspec, 1) ) * np.ones( (1, nv) )
    elif type(noise)== float:
        #thres = np.sqrt(nv)*noise
        #thres_a = np.ones( nspec )*thres 
        sigma_a = np.ones( (nspec, nv) )*noise
    elif type(noise)==np.ndarray:
        # use sigma^2 = Sum_i sigma_i^2
        if noise.ndim==1: 
            #thres = np.sqrt( np.nansum( noise**2 )  )
            #thres_a = np.ones( nspec )*thres
            sigma_a = np.stack( [noise]*nspec )
        elif noise.ndim==2:
            #thres_a = np.sqrt( np.nansum( noise**2, axis=1 )  )
            sigma_a = noise
    thres_a = np.zeros(nspec) #

    if 'lya' in method:
        npk_a, vblue_a, vred_a, vsep_a, peak_ratio_a, r2b_a, fcen_a, sigma_red_a, Af_a = np.zeros(nspec), np.zeros(nspec), np.zeros(nspec), np.zeros(nspec), np.zeros(nspec), np.zeros(nspec), np.zeros(nspec), np.zeros(nspec), np.zeros(nspec)
        I_fit_a = np.zeros_like(spectrum_a)
        for i, spectrum, vcen, thres, sigma in zip(range(nspec), spectrum_a, vsys_a, thres_a, sigma_a):
            if lam.ndim == 1:
                lam_a = lam
            elif lam.ndim == 2:
                lam_a = lam[i, :]

            print('spec', i)
            # use different methods for line parameters
            if np.nansum( spectrum )>thres:
                if method=='lya_fitting_complex':
                    npk_a[i], vblue_a[i], vred_a[i], vsep_a[i], peak_ratio_a[i], r2b_a[i], fcen_a[i], sigma_red_a[i], Af_a[i], I_fit_a[i] = cal_lya_line_parameter(lam_a, spectrum, sigma=sigma, verbose=verbose, vcen=vcen)
                elif method=='lya_fitting_single':
                    npk_a[i], vblue_a[i], vred_a[i], vsep_a[i], peak_ratio_a[i], r2b_a[i], fcen_a[i], sigma_red_a[i], Af_a[i], I_fit_a[i] = cal_lya_line_parameter_single(lam_a, spectrum, sigma=sigma, verbose=verbose, vcen=vcen)
                else:
                    raise Exception('no such method!')
            else:
                print('the spectrum does not exceed the threshold' )
                npk_a[i], vblue_a[i], vred_a[i], vsep_a[i], peak_ratio_a[i], r2b_a[i], fcen_a[i], sigma_red_a[i], Af_a[i], I_fit_a[i] = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        if op in ['std', 'standard']:
            return npk_a, vsep_a, peak_ratio_a, r2b_a, fcen_a, sigma_red_a, I_fit_a
        elif op=='more':
            return npk_a, vblue_a, vred_a, vsep_a, peak_ratio_a, r2b_a, fcen_a, sigma_red_a, I_fit_a
        elif op=='more2':
            return npk_a, vblue_a, vred_a, vsep_a, peak_ratio_a, r2b_a, fcen_a, sigma_red_a, Af_a, I_fit_a
        


    
def peak_stat_vor_map(cube, image, noise_image, noise_cube, size, target_sn, v, vsys, sanity_dict = {'check':False, 'fig_name':None}, op='std' ):
    '''
    Using voronoi mesh method described in Cappellari & Copin (2003)
    Input:
        - cube
        - image
        - noise_image
        - noise_cube: currently unused
    '''    
    signal = image.flatten()
    noise = noise_image.flatten()
    
    # put a non-zero minimum noise 
    min_noise = np.min( noise[noise!=0] )
    noise[noise==0] = min_noise
    
    image_npix = np.shape( image )[0]
    info = {'nx':image_npix, 'ny':image_npix}
    x_aa, y_aa = num_util.create_coord(info, size=size, ndim=2, center=True)
    x_aa = x_aa.flatten()
    y_aa = y_aa.flatten()
    pixelsize = size/image_npix
        
    bin_number, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(
        x_aa, y_aa, signal, noise, target_sn, cvt=True, pixelsize=pixelsize, plot=False,
        quiet=True, sn_func=None, wvt=True)

    # construct voronoi ppv
    n_bin = len(sn)
    spec_npix = len(v)
    ppv_vor = np.zeros( (n_bin, spec_npix) )
    #ppv_noise_vor = np.zeros( (n_bin, spec_npix) )
    img_vor = np.zeros( n_bin )

    for i in range(n_bin):
        p_ix, p_iy = np.where( bin_number.reshape(image_npix, image_npix)==i ) # index of pixel in vor bin
        img_vor[i] = np.mean( image[p_ix, p_iy] )
        ppv_vor[i] = np.mean( cube[p_ix, p_iy, :], axis=0 )
        #ppv_noise_vor[i] = np.sqrt( np.nansum( noise_cube[p_ix, p_iy, :]**2, axis=0 ) ) / nPixels[i]

    vsys_a = np.ones( n_bin )* vsys

    npk_a, vblue_a, vred_a, vsep_a, peak_ratio_a, r2b_a, fcen_a, sigma_red_a, I_fit_a = \
      peak_analysis(ppv_vor, v, vsys_a=vsys_a, verbose=False, method='fitting_complex', noise=None, op='more' )

    if sanity_dict['check']:
        N_check = 100
        nrow_sc = 10
        ncol_sc = 10
        fig_sc, axs_sc = plt.subplots(nrow_sc, ncol_sc, figsize=(ncol_sc*5, nrow_sc*5), sharex=True, sharey=True )
        idx_vb_a = np.flip( np.argsort( img_vor ) )[:: int(n_bin/N_check) ]
        for i_check, idx_vb in zip(range(N_check), idx_vb_a ):
            ip,jp = np.unravel_index( i_check, (nrow_sc,ncol_sc) ) 
            norm = np.max( ppv_vor[idx_vb] )
            axs_sc[ip][jp].plot(v, ppv_vor[idx_vb]/norm )
            #axs_sc[ip][jp].fill_between(v, (ppv_vor[idx_vb]- ppv_noise_vor[idx_vb] )/norm, (ppv_vor[idx_vb]+ppv_noise_vor[idx_vb])/norm  )
            axs_sc[ip][jp].plot(v, I_fit_a[idx_vb]/norm )
            axs_sc[ip][jp].set_ylim(0, 1.2)
            
        fig_sc.savefig(sanity_dict['fig_name'] )
        
        

    lya_img = plotbin.display_bins.display_bins(x_aa, y_aa, bin_number, img_vor, pixelsize=pixelsize, \
                                                 cmap='magma', colorbar=True )
    vblue_img = plotbin.display_bins.display_bins(x_aa, y_aa, bin_number, vblue_a, pixelsize=pixelsize, \
                                                 cmap='magma', colorbar=True )
    vred_img = plotbin.display_bins.display_bins(x_aa, y_aa, bin_number, vred_a, pixelsize=pixelsize, \
                                                 cmap='magma', colorbar=True )
    vsep_img = plotbin.display_bins.display_bins(x_aa, y_aa, bin_number, vsep_a, pixelsize=pixelsize, \
                                                 cmap='magma', colorbar=True )
    r2b_img = plotbin.display_bins.display_bins(x_aa, y_aa, bin_number, r2b_a, pixelsize=pixelsize, \
                                                 cmap='RdBu_r', colorbar=True )
    sigma_red_img = plotbin.display_bins.display_bins(x_aa, y_aa, bin_number, sigma_red_a, pixelsize=pixelsize, \
                                                 cmap='magma', colorbar=True )
    
    lya_img      = np.rot90( lya_img.get_array()    , k=-1)
    vblue_img    = np.rot90( vblue_img.get_array()  , k=-1)
    vred_img     = np.rot90(vred_img.get_array()    , k=-1)
    vsep_img     = np.rot90(vsep_img.get_array()    , k=-1)
    r2b_img      = np.rot90(r2b_img.get_array()     , k=-1)
    sigma_red_img = np.rot90(sigma_red_img.get_array(), k=-1)
    
    if op in ['std', 'standard']:
        return lya_img, vsep_img, r2b_img, I_fit_a
    elif op=='more':
        return lya_img, vblue_img, vred_img, vsep_img, r2b_img, sigma_red_img, I_fit_a    

def cal_EW(lam_a, f, f_cont, spec_type='emission', verbose=False):
    '''
    Calculate equivalent width
    Input:
        f [array]:      spectrum
        f_cont [float]: continuum level
    '''
    
    if verbose: print('cal_EW', f_cont, type(f_cont) )
    assert isinstance(f_cont, (float, np.float64) ), 'f_cont should be a float'

    if spec_type=='emission':
        idx = np.where( f>f_cont )[0]
        lam_a = lam_a[idx]
        f = f[idx]
    
    if spec_type=='emission':
        EW = np.trapz( (f-f_cont)/f_cont, lam_a)
    elif spec_type=='absorption':
        EW = np.trapz( (f_cont-f)/f_cont, lam_a)
    return EW

def cal_EW_a(lam_a, f_a, f_cont_a, spec_type='emission', verbose=False):
    nspec = np.shape(f_a)[0]
    EW_a = np.zeros(nspec)
    assert len(f_cont_a)==nspec, 'f_cont_a should have a length equal to number of spectra'
    for i, f, f_cont in zip(range(nspec), f_a, f_cont_a ):
        EW_a[i] = cal_EW(lam_a, f, f_cont, spec_type=spec_type, verbose=verbose)
    return EW_a 






def IGM_damping_wing(v_a, z, model='Keating23', op_type='percentile'):
    '''
    IGM damping wing raw data (median and its uncertainties)
    Input: 
        - v_a: velocity array
        - z: redshift
        - model: physics model, Keating23 model only include z=6, 6.368, 7, 7.444, 8, 9.024, 10
    '''

    if 'Keating' in model:
        # read data
        hubbleparam = 0.678

        if model=='Keating23':
            h = h5py.File('%s/IGM_dw/tau_norvir_z%0.3f_n600.hdf5'%(num_util_data_dir,z), 'r')
            tau_Lya = h['tau_Lya'][...]
            velaxis = h['velaxis'][...]
            dist_H1= h['dist_H1'][...]
            h.close()
        elif model in ['Keating_rapid', 'Keating_gradual']:
            if model=='Keating_rapid':
                f = h5py.File(f'{num_util_data_dir}/IGM_dw/tauH1_rapid_z{z:.3f}.hdf5','r')

            elif model=='Keating_gradual':
                f = h5py.File(f'{num_util_data_dir}/IGM_dw/tauH1_gradual_z{z:.3f}.hdf5','r')
            
            nbins = f.attrs['npix']
            numlos = f.attrs['nspec']
            xHI = f.attrs['xHI']
            redshift = f.attrs['redshift']
            Mhalo = f['Mhalo'][...]
            MUV = f['MUV'][...]
            dist_H1 = f['r_bubble_ckpch'][...]
            tau_Lya = f['tau_H1'][...]
            velaxis = f['velaxis_kms'][...]
            f.close()

        velaxis=-1.*velaxis
        bubblesize = np.median(dist_H1/1e3/hubbleparam) #cMpc


        if op_type=='percentile':
            dw_Lya_med = np.percentile(np.exp(-tau_Lya),50,axis=0)
            dw_Lya_1s_lo = np.percentile(np.exp(-tau_Lya),15.87,axis=0)
            dw_Lya_1s_hi = np.percentile(np.exp(-tau_Lya),84.13,axis=0)
            dw_Lya_2s_lo = np.percentile(np.exp(-tau_Lya),2.28,axis=0)
            dw_Lya_2s_hi = np.percentile(np.exp(-tau_Lya),97.72,axis=0)
            
            # make vel become ascending
            velaxis = np.flip(velaxis)
            dw_Lya_med = np.flip(dw_Lya_med) 
            dw_Lya_1s_lo = np.flip(dw_Lya_1s_lo)
            dw_Lya_1s_hi = np.flip(dw_Lya_1s_hi) 
            dw_Lya_2s_lo = np.flip(dw_Lya_2s_lo)
            dw_Lya_2s_hi = np.flip(dw_Lya_2s_hi)
                
            if v_a is None:
                # original data
                return bubblesize, velaxis, dw_Lya_med, dw_Lya_1s_lo, dw_Lya_1s_hi, dw_Lya_2s_lo, dw_Lya_2s_hi
            else:
                # interpolation
                f = interpolate.interp1d(velaxis, dw_Lya_med)
                dw_Lya_med = f(v_a)
                f = interpolate.interp1d(velaxis, dw_Lya_1s_lo) 
                dw_Lya_1s_lo = f(v_a)
                f = interpolate.interp1d(velaxis, dw_Lya_1s_hi) 
                dw_Lya_1s_hi = f(v_a)
                f = interpolate.interp1d(velaxis, dw_Lya_2s_lo) 
                dw_Lya_2s_lo = f(v_a)
                f = interpolate.interp1d(velaxis, dw_Lya_2s_hi) 
                dw_Lya_2s_hi = f(v_a)  
                
                return bubblesize, dw_Lya_med, dw_Lya_1s_lo, dw_Lya_1s_hi, dw_Lya_2s_lo, dw_Lya_2s_hi
        elif op_type=='all_los':
            dw_Lya = np.exp(-tau_Lya) # (los, v)

            # make vel ascending
            velaxis = np.flip(velaxis)
            dw_Lya  = np.flip(dw_Lya, axis=1)  

            if v_a is None:
                return velaxis, dw_Lya
            else:
                f = interpolate.interp1d(velaxis, dw_Lya, axis=1)
                dw_Lya = f(v_a)
                return dw_Lya    
                    
    else:
        raise Exception('model not included')

def IGM_damping_wing_rbs(model='Keating23'):
    '''
    IGM damping wing 2d interpolation (redshift, v)
    Input: 
        - model: physics model
    Output:
    '''
    if 'Keating' in model:
        # read data
        dvar = {}
        sfx_a = ['med', '1s_lo', '1s_hi', '2s_lo', '2s_hi']
        hubbleparam = 0.678
        
        if model=='Keating23':
            z_a = [6, 6.368, 7, 7.444, 8, 9.024, 10]
            v_a = np.linspace(-30000, 24000, 20000)
        elif model in ['Keating_rapid', 'Keating_gradual']:
            z_a = [11.633, 12.589, 13.749]
            v_a = np.linspace(-20000, 9500, 20000)
            
        

        for sfx in sfx_a:
            dvar['dw_Lya_'+sfx+'_aa'] = []
        
        # constructing 2d array dw(z, v)
  
        for z in z_a:
            bubblesize, dvar['dw_Lya_med'], dvar['dw_Lya_1s_lo'], dvar['dw_Lya_1s_hi'], dvar['dw_Lya_2s_lo'], dvar['dw_Lya_2s_hi'] = IGM_damping_wing(v_a, z, model=model, op_type='percentile')

            for sfx in sfx_a: 
                dvar['dw_Lya_'+sfx+'_aa'].append( dvar['dw_Lya_'+sfx] ) 
        
        for sfx in sfx_a:
            dvar['dw_Lya_'+sfx+'_aa'] = np.array(dvar['dw_Lya_'+sfx+'_aa'])
            dvar['rbs_dw_Lya_'+sfx] = RectBivariateSpline(z_a, v_a, dvar['dw_Lya_'+sfx+'_aa'], kx=1, ky=3)
            
        return dvar['rbs_dw_Lya_med'], dvar['rbs_dw_Lya_1s_lo'], dvar['rbs_dw_Lya_1s_hi'], dvar['rbs_dw_Lya_2s_lo'], dvar['rbs_dw_Lya_2s_hi']
    else:
        raise Exception('model not included')
    
############################
# measurement related util # 
############################ 

def cal_pi_cross_section_hydrogenic(nu, Z=1, approx=False):
    '''
    calculate photoionization cross section for hydrogenic species
    chapter 13 of Draine 2011
    '''
    I_H = 13.6*u.eV
    sigma_0 = 6.304e-18*Z**(-2)*u.cm**2
    
    if c.h*nu>=Z**2*I_H:
        if approx==False:
            x = np.sqrt(c.h*nu/(Z**2*I_H) - 1 )
            sigma_pe = sigma_0 * (Z**2*I_H/ (c.h*nu) )**4 * np.exp( 4 - 4*np.arctan(x)/x )/(1 - np.exp(-2*np.pi/x) )
        else:
            if c.h*nu<10**2*Z**2*I_H:
                sigma_pe = sigma_0 * (c.h*nu/(Z**2*I_H) )**-3
            elif c.h*nu>1e4*Z**2*I_H:
                sigma_pe = 2**8/(3*Z**2) * c.alpha*np.pi*c.a0**2 * (c.h*nu/(Z**2*I_H) )**-3.5
            else:
                raise Exception('nu not in range of the approximation regime')    
    else:
        sigma_pe=0
        
    return sigma_pe 
        

'''
The method below try to mimic the observational measurement of column density from absorption line
'''

def measure_col_dens_simple(N_a, sigma, verbose=False ):
    '''
    exp(- N_measure sigma ) = < exp( - N sigma) >
    '''
    m = np.mean( np.exp(-N_a*sigma) )
    if m!=0:
        N_measure = -1/sigma * np.log( m )
    elif m==0:
        N_measure = np.min( N_a )
        if verbose: print(f'astro_util.measure_col_dens_simple: the column density is very large so we use the minimum value {N_measure} as N_Measure')
    return N_measure

def measure_col_dens_picket_fence(N_map, lam_a, cal_sigma, log=False ):
    '''
    picket-fence model Heckman+11
    Input:
        N_map, 2D column density map; lam_a, 1d wavelength array; cal_sigma, python function to calculate sigma
    Output:
    '''
    sigma_a = cal_sigma(lam_a)
    # mock absorption spectra
    Irel_a = np.mean(  np.multiply.outer(sigma_a, N_map) , axis=(1,2) )

    def Irel_pf(lam, f_cov, log10_Ncol):
        Ncol=10*log10_Ncol
        sigma=cal_sigma(lam)
        I = f_cov*np.exp(-Ncol* sigma) + (1-f_cov)
        return I

    def log10_Irel_pf(lam, f_cov, log10_Ncol):
        Ncol=10*log10_Ncol
        sigma=cal_sigma(lam)
        I = f_cov*np.exp(-Ncol* sigma) + (1-f_cov)
        logI = np.log10(I)
        return logI
    
    if log==True:
        # Note that there will be significant amount of noise at I ~ 0
        thres=1e-4
        lam_a = lam_a[Irel_a>thres]
        Irel_a = Irel_a[Irel_a>thres]
        popt, pcov = curve_fit(log10_Irel_pf, lam_a, np.log10(Irel_a), bounds=([0, 16], [1, 24])  ) 
    else:
        popt, pcov = curve_fit(Irel_pf, lam_a, Irel_a, bounds=([0, 16], [1, 24]) )
    f_cov, log10_Ncol = popt
    Ncol = 10**log10_Ncol
    
    return f_cov, Ncol





