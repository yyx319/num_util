import os
import sys
from cProfile import label
from lib2to3.pygram import python_grammar_no_print_statement
import re
import numpy as np 
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import scipy 
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from scipy import stats
from scipy.interpolate import interp1d
from scipy import signal
from scipy.io import FortranFile as ff

import astropy.units as u; import astropy.constants as c 
from astropy.cosmology import WMAP7

sys.path.append('../')
import anal_sim_util 
import astro_util
import coord_util
from const_util import *


'''
Curve smoothing
General notes:
x, y should be in linear scale
'''
def smooth_interp(x, y, kind='cubic', ref_fac=10, logx=False, logy=False ):
    '''
    Use interpolation method to smooth the curve. Used for sparse data 
    Inputs:
        kind: kind in interp1d
        ref_fac: refining factor that use to increase the resolution of x
    '''
    # deal with nan
    idx = np.where( (x==x) & (y==y) )[0]
    
    if logx:
        x = np.log10(x)
    if logy:
        y = np.log10(y)
        
    f = interp1d(x[idx], y[idx], kind=kind)
    x_intp = np.linspace(x[0], x[-1], len(x)*ref_fac )
    y_intp = f(x_intp)
    
    if logx:
        x_intp = 10**x_intp 
    if logy:
        y_intp = 10**y_intp 
        
    return x_intp, y_intp

def moving_average(y, method='standard', window_width=10, mode='same', logy=False ):
    '''
    Use moving average method to smooth the curve. Used for dense data
    mode: full, same or valid
    '''
    if logy:
        y = np.log10(y)
    
    if method=='standard':
        window = np.ones(window_width)/window_width
        y_smooth = np.convolve(y, window, mode=mode)        

    elif method=='fast':
        if mode=='same':
            cumsum_vec = np.cumsum(np.insert(y, 0, 0)) 
            y_smooth = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        else:
            raise Exception('other modes not currently included in fast moving average')

    if logy:
        y_smooth = 10**y_smooth
    return y_smooth

def smooth_interp_filter(x, y, kind='cubic', ref_fac=10, fil_dict={'type': 'savgol', 'window':50, 'poly':3}, logx=False, logy=False, yfloor=None ):
    '''
    Used for sparse data
    Hybrid method, first interpolate the curve then filter it. 
    '''
    # remove nan value
    #idx = np.where( (x==x) & (y==y) )[0]
    #x = x[idx]
    #y = y[idx]
    if yfloor is None:
        pass 
    else:
        y[y<yfloor] = yfloor

    if logx:
        x = np.log10(x)
    if logy:
        y = np.log10(y)
        
    x_intp, y_intp = smooth_interp(x, y, kind, ref_fac )
    
    # filter
    if fil_dict['type'] is None:
        y_sm = y_intp
    elif fil_dict['type']=='savgol':
        y_sm = savgol_filter( y_intp, fil_dict['window'], fil_dict['poly'] )
    else:
        raise Exception('Filter type not included!')
    
    if logx:
        x_intp = 10**x_intp 
    if logy:
        y_sm = 10**y_sm 
        
    return x_intp, y_sm 

def smooth_interp_moving_average(x, y, kind='cubic', ref_fac=10, ma_dict={'method':'fast', 'window_width':30, 'mode':'same'}, logx=False, logy=False):
    '''
    Hybrid method, first interpolate then moving average. Used for sparse data
    '''
    if logx:
        x = np.log10(x)
    if logy:
        y = np.log10(y)
    
    x_intp, y_intp = smooth_interp(x, y, kind, ref_fac )
    y_sm = moving_average(y_intp, ma_dict['method'], ma_dict['window_width'], ma_dict['mode'] )
    
    if logx:
        x_intp = 10**x_intp 
    if logy:
        y_sm = 10**y_sm
    return x_intp, y_sm

'''
Image smoothing
'''
def adaptive_smoothing(image, thres_a=None, sig_a=None):
    '''
    This function adaptively smooth the image, i.e. smooth the region with different kernal for different luminosity.
    '''
    
    if thres_a is None and sig_a is None:
        vmax = image.max()
        thres_a = [vmax/1e4, vmax/1e3, vmax/1e2, vmax/1e1]
        sig_a   = [5,2,1,0.5,0.3]
        
    n_subimg = len(sig_a)
    smoothed_image = np.zeros_like( image )
    for i in range(n_subimg):
        sub_img = image.copy()
        if i==0:
            sub_img[sub_img>thres_a[0] ] = 0
        elif i==n_subimg-1:
            sub_img[sub_img<thres_a[-1] ] = 0
        else:
            sub_img[ (sub_img<thres_a[i-1] ) | (sub_img>thres_a[i] ) ] = 0
            
        sub_img_smo = gaussian_filter(sub_img, sig_a[i] )
        smoothed_image = smoothed_image + sub_img_smo
        
    return smoothed_image

def adaptive_smoothing_a(image_a, thres_a=None, sig_a=None ):
    '''
    Perform adaptive smoothing for an array of images
    '''
    smo_img_a = np.zeros_like(image_a)
    for i, image in enumerate(image_a):
        smo_img = adaptive_smoothing(image, thres_a, sig_a)
        smo_img_a[i] = smo_img 
    return smo_img_a  
        
def bilateralFilter(image, vmin, vmax, log, d=-1, sigmaColor=30, sigmaSpace=10):
    import cv2
    image = 256*coord_util.normalize_arr( image, vmin, vmax, log=log )
    image = image.astype('uint8')
    smoothed_image = cv2.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)/256
    smoothed_image = coord_util.rev_normalize_arr(smoothed_image, vmin, vmax, log=log)
    return smoothed_image


def moving_average_2d():
    '''
    perform moving average in 2 dimensions
    '''
    pass 


def RGB_image( map_list, norm_list ):
    '''
    composite RGB image
    '''
    pass 