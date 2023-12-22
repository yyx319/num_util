import warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import Sersic1D
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy import interpolate
from scipy.interpolate import interp1d
import csv
import h5py


import coord_util
from const_util import *
import math_util
from plot import comp_plot

import vorbin
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

################
### usual util #
################

# doppler effect
# v positive if gas moving away from us
def v2lam(v_a, lam0):
    '''
    v in km/s
    lam_a is in the same unit of lam0
    '''
    if type(v_a)==np.ndarray:
        v_a *= u.km/u.s 
    if type(lam0) == float:
        lam0 *= u.Angstrom    
    lam_a = lam0*(1 + v_a/c.c )
    lam_a = lam_a.value 
    return lam_a
v_to_lam = v2lam

def lam2v(lam_a, lam0, value=True):
    v_a = c.c*( lam_a/lam0 - 1 )
    v_a = v_a.to(u.km/u.s)
    if value:
        v_a = v_a.value
    return v_a
lam_to_v = lam2v

def v2nu(v_a, nu_0):
    if type(v_a)==np.ndarray:
        v_a = v_a*u.km/u.s 
    if type(nu_0)==float:
        nu_0 *= u.Hz
    nu_a = nu_0*(1 - v_a/c.c)
    nu_a = nu_a.to(u.Hz)
    return nu_a
v_to_nu=v2nu

def nu2v(nu_a, nu_0):
    v_a = c.c*(1-nu_a/nu_0)
    v_a = v_a.to(u.km/u.s)
    return v_a   
nu_to_v=nu2v 
     
def lam2nu(lam):
    nu = c.c/lam
    return nu 
lam_to_nu = lam2nu

def nu2lam(nu):
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

def dist_to_arcsec():
    '''
    convert the 
    '''

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
        popt, pcov = curve_fit(math_util.log_Sersic, r_a, np.log10(I_r), bounds=(0, [10*np.max(I_r), np.max(r_a ), 10])  ) 
    else:
        popt, pcov = curve_fit(math_util.Sersic, r_a, I_r)
    A, r_eff, n = popt
    
    if verbose:
        plt.figure()
        print('fitted parameters', A, r_eff, n)
        plt.semilogy(r_a, I_r )
        plt.semilogy(r_a, math_util.Sersic(r_a, A, r_eff, n), label='Sersic' )
        plt.legend()
    return A, r_eff, n

def fit_exp(r_a, I_r, sigma_r=0):
    popt, pcov = curve_fit(math_util.exp, r_a, I_r)
    A, rs = popt
    return A, rs

def fit_dbl_exp(r_a, I_r, sigma_r=0, log=True, verbose=False):
    if log==True:
        r_a = r_a[I_r>0]
        I_r = I_r[I_r>0]
        popt, pcov = curve_fit(math_util.log_double_exp, r_a, np.log10(I_r), bounds=([np.log10( np.max(I_r)/100 ), 0,0,0], [ np.log10( 10*np.max(I_r) ), np.max(r_a), 3*np.max(r_a), np.max(r_a)  ])  )
    elif log==False:
        popt, pcov = curve_fit(math_util.double_exp, r_a, I_r)
    logA1, r1, r2, b = popt
    A1 = 10**logA1
    A2 = A1*np.exp( b/r2 - b/r1 )
    
    if verbose:
        print('fitted parameter', A1, A2, r1, r2, b)
        plt.figure(100)
        if log:
            plt.semilogy(r_a, I_r)
            plt.semilogy(r_a, math_util.double_exp(r_a, A1, r1, r2, b), label='double exponential' )
        else:
            plt.plot(r_a, I_r)
            plt.plot(r_a, math_util.double_exp(r_a, A1, r1, r2, b), label='double power law' )
        plt.legend()
        plt.clf()
    return A1, A2, r1, r2, b

####################
# spectra analysis #
####################
def asymmetric_gaussian(lam_a, A, lam_0_asym, a_asym, d, modify=1 ):
    # Modified eq (2) in Shibuya+14b
    sig_asym = a_asym*(lam_a - lam_0_asym)+d
    
    sig_min = 1e-3*d
    sig_asym[sig_asym<sig_min] = sig_min
        
    if modify==1:
        sig_max = 1e1*d
        sig_asym[sig_asym>sig_max] = sig_max
        f_damp = 1
    elif modify==2:
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
                     [ 1.2, np.percentile(lam_a, 75, ),  +0.5, 0.7] ),
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
                                 [ 1.2,  np.percentile(lam_a, 60, ),  0,   0.7,   \
                                   1.2,  np.percentile(lam_a, 75), 0.5, 0.7] ), \
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
                     [1.2, np.percentile(lam_a, 60),  0,    0.70,    \
                      1.2, np.percentile(lam_a, 70),  0.5,  0.70,    \
                      1.2, np.percentile(lam_a, 80),  0.5,  0.70  ] ),
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

def fit_Lya_profile(lam_a, I_a, sigma=None, verbose=True, AIC_thres=np.inf ):
    '''
    TBW
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

def cal_vsep(npk, *arg):
    if npk==1:
        lam = arg[0]
        v = lam_to_v(lam, lam_lya)
        vsep = 2*np.abs(v)
        return v, vsep
    elif npk==2:
        lam1, lam2 = arg
        v1 = lam_to_v(lam1, lam_lya)
        v2 = lam_to_v(lam2, lam_lya)
        vsep = v2-v1
        return v1, v2, vsep
    elif npk==3:
        A1, lam1, a_1_asym, d1, A2, lam2, a_2_asym, d2, A3, lam3, a_3_asym, d3, vcen, lam_a = arg
        v1 = lam_to_v(lam1, lam_lya)
        v2 = lam_to_v(lam2, lam_lya)
        v3 = lam_to_v(lam3, lam_lya)
        
        f1 = asymmetric_gaussian(lam_a, A1, lam1, a_1_asym, d1)
        f2 = asymmetric_gaussian(lam_a, A2, lam2, a_2_asym, d2)
        f3 = asymmetric_gaussian(lam_a, A3, lam3, a_3_asym, d3)
        
        A1 = np.trapz(f1, lam_a)
        A2 = np.trapz(f2, lam_a)
        A3 = np.trapz(f3, lam_a)
        
        if v1<vcen and v2<vcen and v3>vcen:
            vsep = v3 - (v1*A1+v2*A2)/(A1+A2)
            
        elif v1<vcen and v2>vcen and v3>vcen:
            vsep = (v2*A2+v3*A3)/(A2+A3) - v1
        else:
            print('astro_util.cal_vsep:triple peak on one side')
            v_pk = (v1*A1 + v2*A2 + v3*A3)/(A1+A2+A3)
            vsep = 2* np.abs( v_pk ) 
        return v1, v2, v3, vsep 

def cal_r2b_fcen(lam_a, spectrum, vcen):
    v = lam_to_v(lam_a, lam_lya)
    ridx = v>vcen
    bidx = v<vcen
    cenidx = np.where( np.abs(v-vcen)<40 )
    r2b = np.trapz( spectrum[ridx], v[ridx] ) / np.trapz( spectrum[bidx], v[bidx] )
    fcen = np.trapz( spectrum[cenidx], v[cenidx] ) / np.trapz( spectrum, v )
    return r2b, fcen 

def cal_lya_line_parameter(lam_a, spectrum, sigma=None, verbose=True, vcen=None):
    '''
    calculate
    '''
    npk, par, I_fit = fit_Lya_profile(lam_a, spectrum, sigma=sigma, verbose=verbose)
    if npk==0:
        vsep, peak_ratio, r2b, fcen, FWHM_red = np.nan, np.nan, np.nan, np.nan, np.nan
    elif npk==1:
        A, lam_asym, a_asym, d = par 
        _, vsep = cal_vsep(npk, lam_asym)
        peak_ratio = np.nan
        r2b, fcen = cal_r2b_fcen(lam_a, spectrum, vcen)
        FWHM_red = 3e5*d/lam_lya 
    elif npk==2:
        A1, lam_1_asym, a_1_asym, d1, A2, lam_2_asym, a_2_asym, d2 = par
        _, _, vsep = cal_vsep(npk, lam_1_asym, lam_2_asym)
        peak_ratio = A2/A1
        r2b, fcen = cal_r2b_fcen(lam_a, spectrum, vcen)
        FWHM_red = 3e5*d2/lam_lya 
    elif npk==3:
        A1, lam_1_asym, a_1_asym, d1, A2, lam_2_asym, a_2_asym, d2, A3, lam_3_asym, a_3_asym, d3 = par
        _, _, _, vsep = cal_vsep(npk, A1, lam_1_asym, a_1_asym, d1, A2, lam_2_asym, a_2_asym, d2, A3, lam_3_asym, a_3_asym, d3, vcen, lam_a)
        peak_ratio=A3/A1 
        r2b, fcen = cal_r2b_fcen(lam_a, spectrum, vcen)
        FWHM_red = 3e5*d3/lam_lya 
    
    return npk, vsep, peak_ratio, r2b, fcen, FWHM_red, I_fit

def peak_analysis(spectrum_a, v, vsys_a=None, verbose=False, nv_rebin=None, method='fitting_complex', noise=None):
    '''
    function to calcuate line parameters for Lya double peak spectra
    Input:

    TBD:
        - delete nv_rebin and use moving average algorithm
    '''
    
    nspec = np.shape(spectrum_a)[0]
    nv = len(v)

    lam_a = v_to_lam(v, lam_lya )
           
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

    if method=='fitting_double':
        peak_vel_blue_a, peak_vel_red_a, FWHM_blue_a, FWHM_red_a, ablue_a, ared_a, vsep_a, peak_ratio_a, r2b_a, fcen_a = [np.zeros(nspec), np.zeros(nspec), np.zeros(nspec), np.zeros(nspec), np.zeros(nspec), np.zeros(nspec), np.zeros(nspec), np.zeros(nspec), np.zeros(nspec), np.zeros(nspec)]
        I_fit_a = np.zeros_like(spectrum_a)
        for i, spectrum, vcen, thres, sigma in zip(range(nspec), spectrum_a, vsys_a, thres_a, sigma_a):
            if np.nansum( spectrum )>thres:
                try:
                    A1, lam1_asym, a1_asym, d1, A2, lam2_asym, a2_asym, d2, fitted, _ = fit_double_asymmetric_gaussian(lam_a, spectrum, sigma=sigma, verbose=verbose)
                    peak_vel_blue_a[i], peak_vel_red_a[i], vsep_a[i] = cal_vsep(2, lam1_asym, lam2_asym )
                    FWHM_blue_a[i] = 3e5*d1/lam_lya; FWHM_red_a[i] = 3e5*d2/lam_lya 
                    ablue_a[i] = a1_asym; ared_a[i] = a2_asym
                    peak_ratio_a[i] = A2/A1
                    # red/blue ratio and fcen
                    r2b_a[i], fcen_a[i] = cal_r2b_fcen(lam_a, spectrum, vcen=vcen)

                except Exception as e:
                    print('astro_util.peak_analysis, Error:',e)
                    peak_vel_blue_a[i], peak_vel_red_a[i], FWHM_blue_a[i], FWHM_red_a[i], ablue_a[i], ared_a[i], vsep_a[i], peak_ratio_a[i], r2b_a[i], fcen_a[i] = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                    fitted = np.zeros_like(spectrum)
            else:
                print('peak_analysis: the spectrum does not exceed the threshold' )
                peak_vel_blue_a[i], peak_vel_red_a[i], FWHM_blue_a[i], FWHM_red_a[i], ablue_a[i], ared_a[i], vsep_a[i], peak_ratio_a[i], r2b_a[i], fcen_a[i] = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                fitted = np.zeros_like(spectrum)
            I_fit_a[i] = fitted 
        return peak_vel_blue_a, peak_vel_red_a, FWHM_blue_a, FWHM_red_a, ablue_a, ared_a, vsep_a, peak_ratio_a, r2b_a, fcen_a, I_fit_a 
    
    elif method=='fitting_complex':
        npk_a, vsep_a, peak_ratio_a, r2b_a, fcen_a, FWHM_red_a = np.zeros(nspec), np.zeros(nspec), np.zeros(nspec), np.zeros(nspec), np.zeros(nspec), np.zeros(nspec)
        I_fit_a = np.zeros_like(spectrum_a)
        for i, spectrum, vcen, thres, sigma in zip(range(nspec), spectrum_a, vsys_a, thres_a, sigma_a):
            if np.nansum( spectrum )>thres:
                npk_a[i], vsep_a[i], peak_ratio_a[i], r2b_a[i], fcen_a[i], FWHM_red_a[i], I_fit_a[i] = cal_lya_line_parameter(lam_a, spectrum, sigma=sigma, verbose=verbose, vcen=vcen)
            else:
                print('peak_analysis: the spectrum does not exceed the threshold' )
                npk_a[i], vsep_a[i], peak_ratio_a[i], r2b_a[i], fcen_a[i], FWHM_red_a[i], I_fit_a[i] = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        return npk_a, vsep_a, peak_ratio_a, r2b_a, fcen_a, FWHM_red_a, I_fit_a


def peak_stat_map(cube, v, vsys, res=10, nv_rebin=None, verbose=False, method='fitting_double', noise=0):
    '''
    spatially resolved Lya profile map (cartesian)
    Input: 
    Output:
    '''
    if nv_rebin==None:
        nv_rebin = len(v)
    cube_rebin = coord_util.rebin(cube, res, res, nv_rebin )
    v = coord_util.rebin(v, nv_rebin)
    spectrum_a = cube_rebin.reshape(res**2, nv_rebin )

    vsys_a = vsys*np.ones(res**2)

    if method=='fitting_double':
        peak_vel_blue_a, peak_vel_red_a, FWHM_blue_a, FWHM_red_a, ablue_a, ared_a, vsep_a, peak_ratio_a, r2b_a, fcen_a, I_fit_a = peak_analysis(spectrum_a, v, vsys_a, verbose=False, nv_rebin=nv_rebin, method=method, noise=noise, )   

        peak_vel_blue_img = peak_vel_blue_a.reshape(res,res)
        peak_vel_red_img = peak_vel_red_a.reshape(res,res)
        FWHM_blue_img = FWHM_blue_a.reshape(res,res)  
        FWHM_red_img = FWHM_red_a.reshape(res,res)
        ablue_img = ablue_a.reshape(res, res)
        ared_img = ared_a.reshape(res, res)
        vsep_img = vsep_a.reshape(res,res)
        peak_ratio_img = peak_ratio_a.reshape(res,res)
        r2b_img = r2b_a.reshape(res,res)
        fcen_img = fcen_a.reshape(res,res)
        I_fit_aa = I_fit_a.reshape(res, res, nv_rebin)
        return cube_rebin, peak_vel_blue_img, peak_vel_red_img, FWHM_blue_img, FWHM_red_img, ablue_img, ared_img, vsep_img, peak_ratio_img, r2b_img, fcen_img, I_fit_aa 
    
    elif method=='fitting_complex':
        npk_a, vsep_a, peak_ratio_a, r2b_a, fcen_a, FWHM_red_a, I_fit_a = peak_analysis(spectrum_a, v, vsys_a, verbose=False, nv_rebin=nv_rebin, method=method, noise=noise) 
        npk_img = npk_a.reshape(res, res)  
        vsep_img = vsep_a.reshape(res, res)
        peak_ratio_img = peak_ratio_a.reshape(res, res)
        r2b_img = r2b_a.reshape(res, res)
        fcen_img = fcen_a.reshape(res, res)
        FWHM_red_img = FWHM_red_a.reshape(res, res)
        I_fit_aa = I_fit_a.reshape(res, res, nv_rebin)
        return cube_rebin, npk_img, vsep_img, peak_ratio_img, r2b_img, fcen_img, FWHM_red_img, I_fit_aa
        

def peak_stat_vor_map(cube, image, noise_image, noise_cube, size, target_sn, v, vsys, sanity_dict = {'check':False, 'fig_name':None} ):
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
    x_aa, y_aa = coord_util.create_coord(info, size=size, ndim=2, center=True)
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

    npk_a, vsep_a, peak_ratio_a, r2b_a, fcen_a, FWHM_red_a, I_fit_a = \
      peak_analysis(ppv_vor, v, vsys_a=vsys_a, verbose=False, nv_rebin=spec_npix, method='fitting_complex', noise=None )

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
    vsep_img = plotbin.display_bins.display_bins(x_aa, y_aa, bin_number, vsep_a, pixelsize=pixelsize, \
                                                 cmap='magma', colorbar=True )
    r2b_img = plotbin.display_bins.display_bins(x_aa, y_aa, bin_number, r2b_a, pixelsize=pixelsize, \
                                                 cmap='RdBu_r', colorbar=True )
    
    lya_img = lya_img.get_array()
    vsep_img = vsep_img.get_array()
    r2b_img  = r2b_img.get_array()
    
    return lya_img, vsep_img, r2b_img, I_fit_a
    

def cal_EW(lam_a, f, f_cont, spec_type='emission'):
    if spec_type=='emission':
        EW = np.trapz( (f-f_cont)/f_cont, lam_a)
    elif spec_type=='absorption':
        EW = np.trapz( (f_cont-f)/f_cont, lam_a)
    return EW

def cal_EW_a(lam_a, f_a, f_cont_a, spec_type='emission'):
    nspec = np.shape(f_a)[0]
    EW_a = np.zeros(nspec)
    for i, f, f_cont in zip(range( len(f_a) ), f_a, f_cont_a ):
        EW_a[i] = cal_EW(lam_a, f, f_cont, spec_type='emission')
    return EW_a 

def IGM_damping_wing(v_a, z, model='Keating23'):
    '''
    IGM damping wing
    Input: 
        - v_a: velocity array
        - z: redshift
        - model: physics model, Keating23 model only include z=6, 6.368, 7, 7.444, 8 
    Output:
    '''
    if model=='Keating23':
        # read data
        hubbleparam = 0.678

        h = h5py.File('data/IGM_dw/tau_norvir_z%0.3f_n600.hdf5'%z, 'r')
        tau_Lya = h['tau_Lya'][...]
        velaxis = h['velaxis'][...]
        dist_H1= h['dist_H1'][...]
        h.close()

        bubblesize = np.median(dist_H1/1e3/hubbleparam) #cMpc

        dw_Lya_med = np.percentile(np.exp(-tau_Lya),50,axis=0)
        dw_Lya_1s_lo = np.percentile(np.exp(-tau_Lya),15.87,axis=0)
        dw_Lya_1s_hi = np.percentile(np.exp(-tau_Lya),84.13,axis=0)
        dw_Lya_2s_lo = np.percentile(np.exp(-tau_Lya),2.28,axis=0)
        dw_Lya_2s_hi = np.percentile(np.exp(-tau_Lya),97.72,axis=0)
        velaxis = -1.*velaxis

        # interpolation
        f = interpolate.interp1d(velaxis, dw_Lya_med)
        T_med_a = f(v_a)
        f = interpolate.interp1d(velaxis, dw_Lya_1s_lo) 
        T_1s_lo_a = f(v_a)
        f = interpolate.interp1d(velaxis, dw_Lya_1s_hi) 
        T_1s_hi_a = f(v_a)
        f = interpolate.interp1d(velaxis, dw_Lya_2s_lo) 
        T_2s_lo_a = f(v_a)
        f = interpolate.interp1d(velaxis, dw_Lya_2s_hi) 
        T_2s_hi_a = f(v_a)
        
    return bubblesize, T_med_a, T_1s_lo_a, T_1s_hi_a, T_2s_lo_a, T_2s_hi_a


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

def measure_col_dens_simple(N_a, sigma ):
    '''
    exp(- N_measure sigma ) = < exp( - N sigma) >
    '''
    m = np.mean( np.exp(-N_a*sigma) )
    if m!=0:
        N_measure = -1/sigma * np.log( m )
    elif m==0:
        print('astro_util.measure_col_dens_simple: the column density is very large so we use minimum value.')
        N_measure = np.min( N_a )
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





