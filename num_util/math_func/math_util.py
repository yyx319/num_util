'''
math util
'''
import numpy as np
import math 
import random
import types

import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline


from num_util.const_util import *

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


'''
Statistics
'''
def N_fac_pdf(PDF, x_a):
    '''
    Return the normalization factor the PDF (un-normalized)
    Inputs:
        PDF is a function
        x_a is an array
    '''
    if type(PDF)==types.FunctionType:
        N = 1/np.trapz( PDF(x_a), x_a )
    elif type(PDF)==np.ndarray:
        N = 1/np.trapz( PDF, x_a )
    else:
        raise Exception('PDF data type wrong')
    return N 



# normal distribution
def gaussian(x, A, x0, sigma):
    '''
    Gaussian function
    '''
    f = A*np.exp( -(x-x0)**2 / (2*sigma**2) )
    return f

def double_gaussian(x, A1, x0_1, sigma1, A2, x0_2, sigma2):
    f1 = gaussian(x, A1, x0_1, sigma1)
    f2 = gaussian(x, A2, x0_2, sigma2)
    f = f1+f2
    return f

def nor_gaussian(x, x0, sigma):
    '''
    normalized Gaussian function
    '''
    A = 1/( sigma*np.sqrt(2*np.pi) )
    f = gaussian(x, A, x0, sigma)
    return f

# log normal distribution
def log_normal(x, mu, sigma ):
    '''
    log normal distribution for x
    '''
    P = 1/( np.sqrt(2*np.pi)*sigma*x ) * np.exp( - ( np.log(x)-mu )**2/(2*sigma**2) )
    return P

def log_normal_10(x, mu, sigma):
    '''
    Alternative form of log normal distribution 
    '''
    P = 1/( np.log(10)*np.sqrt(2*np.pi)*sigma*x ) * np.exp( - ( np.log10(x)-mu )**2/(2*sigma**2) )
    return P
    
def truncated_lognormal(x, mu, sigma, domain):
    '''
    if the domain of x is 0->inf, then A=1
    '''
    pass 

# other
def cal_FWHM(par, dist='gaussian'):
    '''
    Calculate FWHM from the parameter of the distribution
    '''
    if dist=='gaussian':
        sigma=par 
        FWHM = 2*np.sqrt(2*np.log(2))*sigma # FWHM ~ 2.355*sigma
    else:
        raise Exception('Distribution not included')
    return FWHM

def sample_PDF( x_domain, PDF, N, method='von-newmann' ):
    '''
    Sample N data from PDF(x)
    Input: 
        x_a: array
        PDF: function
    '''
    xmin, xmax = x_domain 
    x_a = np.linspace(xmin, xmax, 1000)
    
    assert isinstance(PDF, types.FunctionType), 'PDF is not a function'

    PDF_max = np.max( PDF(x_a) )    
    x_sam = []
    if method=='von-newmann':
        while 1:
            r = random.random()
            x = r*(xmax-xmin) + xmin 
            P = random.random()*PDF_max
            if P<PDF(x):
                x_sam.append(x)
            if len(x_sam)==N:
                break
    else:
        raise Exception('Method not included')   
    return x_sam


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
    
def are_both_similar( nor, val, thres=1e-2, nor2=1):
    if isinstance(nor, float):
        if nor==0 and val==0:
            return True
        elif nor==0 and val!=0:
            dev = (val-nor)/nor2
        else:
            dev = (val-nor)/nor 
    elif isinstance(nor, np.ndarray):
        s1 = np.sum( (val-nor)**2 ) 
        s2 = np.sum( nor**2 ) 
        if s1==0 and s2==0:
            return True 
        elif s1!=0 and s2==0:
            dev = np.sqrt( s1/nor2**2 )
        else:
            dev = np.sqrt( s1/s2 )
    dev = np.abs(dev)
    
    if dev<=1e-2:
        return True
    else:
        return False
    
def are_they_similar( nor, arr, thres=1e-2, nor2=1 ):
    '''
    Inputs:
        arr: 
    '''
    similar_a = []
    for elm in arr:        
        similar = are_both_similar( nor, elm, thres, nor)
        similar_a.append(similar)
    return similar_a 