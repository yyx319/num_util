#############
# dust util #
#############
import os
import sys
sys.path.append('../')
import astropy.constants as c
import astropy.units as u

from num_util.const_util import *


'''
Chapter 21 in Draine 2011
Gnedin08
'''

# Table 21.1 in Draine 2011
band_tab = {
    'V': 0.5470*u.micrometer,
    'B': 0.4405*u.micrometer
}
    
def dust_cross_sec(lam, paper='Gnedin08', model='SMC'):
    '''
    Calculate dust cross section
    '''    
    lam = lam.to(u.micrometer).value # lam should be in micrometer
    
    if paper=='Gnedin08':
        def cal_f(x, a, b, p, q):
            f = a/( x**p + x**(-q) + b )
            return f
        if model=='SMC':
            sigma_0 = 1.0e-22
            # fitting parameter
            lam_i_a = [0.042, 0.08, 0.22,  9.7,   18,    25,    0.067]
            a_a     = [185,   27,   0.005, 0.010, 0.012, 0.030, 10   ]
            b_a     = [90,    15.5, -1.95, -1.95, -1.8,  0,     1.9  ]
            p_a     = [2,     4,    2,     2,     2,     2,     4    ]
            q_a     = [2,     4,    2,     2,     2,     2,     15   ]
        elif model=='LMC':
            sigma_0 = 3.0e-22 
            lam_i_a = [0.046, 0.08, 0.22,  9.7,   18,    25,   0.067]
            a_a     = [90,    19,   0.023, 0.005, 0.006, 0.02, 10   ]
            b_a     = [90,    21,   -1.95, -1.95, -1.8,  0,    1.9  ]
            p_a     = [2,     4.5,  2,     2,     2,     2,    4    ] 
            q_a     = [2,     4.5,  2,     2,     2,     2,    15   ]

        sum_f = 0
        for i, lam_i, ai, bi, pi, qi in zip(range(7), lam_i_a, a_a, b_a, p_a, q_a ):
            sum_f += cal_f(lam/lam_i, ai, bi, pi, qi )
        sigma_d = sigma_0*sum_f
    return sigma_d

def dust_cross_sec_band(band, paper='Gnedin08', model='SMC'):
    lam = band_tab[band]
    sigma_d = dust_cross_sec(lam, paper, model)
    return sigma_d 

def cal_Alam( N, lam ):
    '''
    calculate dust extinction
    Inputs:
        - N: dust column density
        - lam: wavelength
    '''
    sigma_d = dust_cross_sec(lam)
    tau_lam = N*sigma_d
    Alam = 1.086*tau_lam
    return Alam

def cal_Aband( N, band ):
    '''
    calculate Alam in a particular band, such as AV
    '''
    lam = band_tab[band]
            
    Aband = cal_Alam(N, lam)
    return Aband 

def cal_reddening(N):
    '''
    calculated the reddening E(B-V) 
    Input:
        N: dust column density
    '''
    AB = cal_Aband( N, 'B' )
    AV = cal_Aband( N, 'V' )
    E_B_V = AB-AV
    return E_B_V 

def cal_RV(paper='Gnedin08', model='SMC'):
    '''
    RV = AV/E(B-V) = sigma_V/(sigma_B-sigma_V)
    '''
    sigma_V = dust_cross_sec_band('V', paper, model)
    sigma_B = dust_cross_sec_band('B', paper, model)
    RV = sigma_V/(sigma_B-sigma_V)
    return RV 


def cal_ndust_Laursen09(Z, nHI, nHII, Z_0=0.005, f_ion=0.01):
    '''
    Calculate dust density according to Laursen09 (also refer to Michel-Dansac20)
    '''
    ndust = Z/Z_0*(nHI+f_ion*nHII)
    return ndust