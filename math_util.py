'''
math util
'''
import numpy as np
import math 
from astropy.modeling.models import Sersic1D
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

import coord_util
from const_util import *

'''
basic function
'''
def linear_two_points(x, x1, x2, y1, y2):
    y = (y2-y1)/(x2-x1) * (x-x1) + y1 
    return y

def power_law_two_points(x, x1, x2, y1, y2):
    p = np.log10(y2/y1) / np.log10(x2/x1)
    y = y1*(x/x1)**p
    return y

def exp(r, A, rs):
    f = A*np.exp(r/rs)
    return f

def gaussian(x, A, x0, sigma):
    '''
    Gaussian function
    '''
    f = A*np.exp( -(x-x0)**2 / (2*sigma**2) )
    return f

def nor_gaussian(x, x0, sigma):
    '''
    normalized Gaussian function
    '''
    A = 1/( sigma*np.sqrt(2*np.pi) )
    f = gaussian(x, A, x0, sigma)
    return f

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
functions for array 
'''
def find_nearest(array, value, sorted=False):
    '''
    Find the number in array that is nearest to value.
    '''
    if sorted==False:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    else:
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return array[idx-1]
        else:
            return array[idx]
        
        
'''
Auto correlation function (ACF)
'''

def ACF( L, Norm='divide_ACF_at_centre' ):
    fL = np.fft.fftn(L, s=None, axes=None)
    psi = fL*np.conj(fL)
    A3 = np.fft.ifftn(psi)
    
    nx, ny, nz = np.shape(L)
    A3 = np.real(A3)/(nx*ny*nz)
    
    if Norm=='divide_ACF_at_centre':
        A3_nor = A3/A3[0,0,0]
    else:
        raise Exception('Normalization not included')
    A3_nor = np.roll(A3_nor, [ int(nx/2), int(ny/2), int(nz/2)], axis=[0,1,2])
    
    return 
    