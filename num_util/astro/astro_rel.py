import numpy as np
from astropy.modeling.models import Sersic1D

from num_util.const_util import *

import num_util 

'''
functions in astrophysics
'''
# double exponential profile fitting 
def double_exp(r, A1, r1, r2, b):
    f1 = A1*np.exp(-r/r1)
    A2 = A1*np.exp( b/r2 - b/r1 )
    f2 = A2*np.exp(-r/r2)
    f = ( 1-np.heaviside(r-b, 0) )*f1 + np.heaviside(r-b, 0)*f2 
    return f

def log_double_exp(r, log_A1, r1, r2, b):
    A1 = 10**log_A1
    f = double_exp(r, A1, r1, r2, b)
    f = np.log10(f)
    return f

'''
BT Galactic dynamics
'''
# Appendix A BT
h7 = 1.05 

# Sersic fit 
# eq (1.17) in BT
# https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Sersic1D.html
def Sersic(r, A, r_eff, n):
    s = Sersic1D(amplitude=A, r_eff=r_eff, n=n)
    f = s(r)
    return f 

def log_Sersic(r, A, r_eff, n):
    s = Sersic1D(amplitude=A, r_eff=r_eff, n=n)
    f = np.log10( s(r) )
    return f 

def Schechter_law(L, phi_s, L_s, alpha):
    '''
    eq (1.18) in BT
    '''
    phi_L = phi_s * (L/L_s)**alpha * np.exp(-L/L_s) / L_s
    return phi_L 

def UV_Schechter_law(MUV, phi_s, MUV_s, alpha):
    '''
    section 4.2 in Bouwens+2015
    '''
    fac = 10**( -0.4*(MUV-MUV_s) )
    phi_L = phi_s * ( np.log(10)/2.5 ) * fac**(alpha+1) * np.exp(-fac) 
    return phi_L

def fundamental_plane(sigma_para, mean_Ie, C):
    '''
    eq (1.20) in BT
    '''
    raise Exception('Unit to be implemented in this function')
    log10Re = 1.24*np.log10(sigma_para) - 0.82*np.log10(mean_Ie) + C
    Re = 10**log10Re
    return Re 

def Faber_Jackson_law(LR):
    '''
    eq (1.21) in BT
    '''
    log10_sigma_para_150 = 0.25*np.log10( LR/(1e10*h7**-2*c.Lsun) )
    
    sigma_para_150 = 10**log10_sigma_para_150
    sigma_para = sigma_para_150 * 150*u.km/u.s 
    
    return sigma_para 
    
def Kormendy_relation(Re):
    '''
    eq (1.22a) in BT 
    '''
    log10_IeR_1p2 = -0.8*np.log10( Re/(h7**-1*u.kpc) )
    
def Tully_Fisher_law(v_c):
    '''
    eq (1.24) in BT
    '''
    log10_LR_10 = 3.5*np.log10(v_c/(200*u.km/u.s) ) + 0.5
    LR_10 = 10**log10_LR_10
    LR = (LR_10*1e10*u.Lsun )/ h7**2
    
'''
Mason 18
'''
def M18_V(Mh):
    m = 0.32 
    c1 = 2.48 # avoid overwriting c (astropy.constants)
    V = m*np.log10( Mh/(1.55e12*u.Msun) ) + c1
    return V

def M18_p_vred(vred, Mh):
    '''
    Mason et al 2018 eq (1)
    '''
    V = M18_V(Mh)
    sigma_v = 0.24
    P = num_util.log_normal_10(vred, V, sigma_v)
    return P 

    
'''
DW12 and Weinberger+19
'''

def cal_P_REW_MUV(REW_a, MUV, z, verbose=True):
    assert type(z)== float
    REW_c = 23 + 7*(MUV+21.9) + 6*(z-4)
    REW_min = np.piecewise(MUV, [MUV<-21.5, MUV>-19, (-21.5<=MUV) & (MUV<=-19)           ], 
                                [-20,       17.5,    lambda MUV: -20 + 6*(MUV+21.5)**2   ])         # this parameter is REW_min in W19 is -a_1 in DW2012  

    REW_max=300

    N = 1/REW_c * ( np.exp(-REW_min/REW_c) - np.exp(-REW_max/REW_c) )**(-1) 
        
    P_REW_MUV = N*np.exp(-REW_a/REW_c)
    P_REW_MUV = np.where( (REW_min<REW_a) & (REW_a<REW_max), P_REW_MUV, 0)
                
    return P_REW_MUV


    
def cal_PEW_z_unnor(REW_a, z, verbose=True, paper='W19'):
    assert type(z)== float
    assert paper in ['DW12', 'W19']
    
    beta=-1.7
    lam_lya = 1216
    nu_lya = (c.c/(lam_lya*u.Angstrom) ).to(u.Hz).value
    lam_UV = 1700
    C = nu_lya/lam_lya * (lam_UV/lam_lya)**(-beta-2)

    if paper=='DW12':
        # Table 2 in DW12
        if z==3.1:   La_min=2e42; La_max=50e42
        elif z==3.7: La_min=4e42; La_max=40e42
        elif z==5.7: La_min=2.5e42; La_max=40e42
    elif paper=='W19':
        if z==5.7:   La_min=6.3e42 
        elif z==6.6: La_min=7.9e42
        elif z==7.0: La_min=2.0e42
        elif z==7.3: La_min=2.4e42
        La_max=40e42
        
    N_La = 301
    La = np.logspace( np.log10(La_min), np.log10(La_max), N_La )
    log_La = np.log10(La)
    MUVc = -2.5*np.log10( np.outer( La, 1/(C*REW_a) ) ) + 51.6
    
    P_REW_MUVc_z = cal_P_REW_MUV( np.stack([REW_a]*N_La ), MUVc, z)
    
    if paper == 'DW12':
        if z==3.1:   phi_s = 1.7; alpha = -1.73
        elif z==3.7: phi_s = 1.3; alpha = -1.73
        elif z==5.7: phi_s=1.4 ;  alpha = -1.74
        if 3<z and z<6: MUV_s = -21.02+0.36*(z-3.8)
    elif paper == 'W19':
        # adopt Bouwens 15 (B15) UVLF
        # Table 6 of B15
        if z in [6.6, 7.0]:
            # z 6.8 in All Fields 
            MUV_s = -20.87; phi_s = 0.29; alpha = -2.06 
        elif z==7.3:
            # z 7.9 in ALL Fields
            MUV_s = -20.63; phi_s = 0.21; alpha = -2.02
            
                    
    phi_MUVc_z = UV_Schechter_law(MUVc, phi_s, MUV_s, alpha)
    PEW_z_unnor = np.trapz( P_REW_MUVc_z*phi_MUVc_z, log_La, axis=0)
    return PEW_z_unnor

def DW12_PEW(REW_a,z, verbose=True):    
    # calculate the normalization constant
    assert type(z)== float
    N_REW_full = 1001
    REW_full_a = np.linspace(1, 300, N_REW_full)
    N = num_util.N_fac_pdf(cal_PEW_z_unnor(REW_full_a, z, verbose=verbose), REW_full_a)
    if verbose: print('N', N)
    
    PEW_z = N*cal_PEW_z_unnor(REW_a, z, verbose=verbose)
    
    return PEW_z 

