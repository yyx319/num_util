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

import num_util
from num_util.const_util import *


'''
Curve smoothing
General notes:
x, y should be in linear scale
'''
def smooth_interp(x, y, kind='linear', ref_fac=10, logx=False, logy=False, nan_dealer={'method': 'remove'}, inf_dealer='remove', remove_close_x=True, verbose=True ):
    '''
    Use interpolation method to smooth the curve. Used for sparse data 
    Inputs:
        kind: kind in interp1d
        ref_fac: refining factor that use to increase the resolution of x
    '''
    assert len(x)==len(y) # x and y should have the same length
    
    # deal with nan
    if nan_dealer['method']=='remove':
        idx = np.where( (x==x) & (y==y) )[0]
        x=x[idx]
        y=y[idx]
    elif nan_dealer['method']=='yfloor':
        ymin = np.nanmin(y)
        if logy:
            if ymin>0:
                yfloor = ymin/1e4
            else:
                ymin_pos = np.nanmin( y[y>0] )
                yfloor = ymin_pos/1e4
                y[y<=0] = yfloor
            y[y!=y] = yfloor
        else:
            if ymin>=0:
                yfloor = ymin/1e4 
            else:
                yfloor = ymin*1e2
            y[y!=y] = yfloor
    elif nan_dealer['method']=='set_yfloor':
        yfloor=nan_dealer['yfloor']
        y[y!=y]=yfloor
    else: raise Exception('nan_dealer method not included!')
    
    # deal with inf
    if inf_dealer=='remove':
        if not logy:
            idx = np.where( np.isfinite(y) )[0]
        elif logy:
            idx = np.where( (y>0) & np.isfinite(y) )[0]
        x=x[idx]
        y=y[idx]
    else: raise Exception('inf_dealer method not included!')
        
    if logx:
        x = np.log10(x)
    if logy:
        y = np.log10(y)

    # deal near points in x
    if remove_close_x: 
        dx = x[1:]-x[:-1]
        dx_mean = np.mean(dx)
        if verbose: print(f'smooth_interp dx {dx}, dx_mean {dx_mean}')
        idx_a = np.where(dx < 0.2*dx_mean )[0]
        if verbose: print(f'smooth_interp: remove close x {idx_a}')
        x = np.delete(x, idx_a)
        y = np.delete(y, idx_a)

    f = interp1d(x, y, kind=kind)
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

def smooth_interp_filter(x, y, kind='cubic', ref_fac=10, fil_dict={'type': 'savgol', 'window':50, 'poly':3}, logx=False, logy=False, nan_dealer={'method': 'remove'}, inf_dealer='remove', verbose=True ):
    '''
    Used for sparse data
    Hybrid method, first interpolate the curve then filter it. 
    '''

    x_intp, y_intp = smooth_interp(x, y, kind, ref_fac, logx=logx, logy=logy, nan_dealer=nan_dealer, inf_dealer=inf_dealer, verbose=verbose )
    
    if logx:
        x_intp = np.log10(x_intp)
    if logy:
        y_intp = np.log10(y_intp)
        
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

def smooth_interp_moving_average(x, y, kind='cubic', ref_fac=10, ma_dict={'method':'fast', 'window_width':30, 'mode':'same'}, logx=False, logy=False, nan_dealer={'method': 'remove'}, inf_dealer='remove'):
    '''
    Hybrid method, first interpolate then moving average. Used for sparse data
    '''
    if logx:
        x = np.log10(x)
    if logy:
        y = np.log10(y)
    
    x_intp, y_intp = smooth_interp(x, y, kind, ref_fac, nan_dealer=nan_dealer, inf_dealer=inf_dealer )
    y_sm = moving_average(y_intp, ma_dict['method'], ma_dict['window_width'], ma_dict['mode'] )
    
    if logx:
        x_intp = 10**x_intp 
    if logy:
        y_sm = 10**y_sm
    return x_intp, y_sm

'''
Image smoothing
'''
def adaptive_smoothing(image, cln_image=None, thres_a=None, sig_a=None, def_type=None):
    '''
    This function adaptively smooth the image. This algorithm first divide the map into different regions with  different intensity or refinement level specified by thres_a. Then it smooths different regions by different kernals specified by sig_a. thres_a and sig_a should be in ascending order. Some predefined thres_a and sig_a can also be specified by def_type.
    Inputs:
        image: 2D array, the image to be smoothed
        cln_image: 2D array showing the clearness of image, if None, use image as cln_image. 
        thres_a: 1D array, the threshold for each sub-image, if None, use default value
        sig_a: 1D array, the sigma for each sub-image, the length of this should be len(thres_a)+1, if None, use default value
        def_type: str, the default type of threshold and sigma. This will only be used if both thres_a and sig_a are None.
    '''
    if np.min(image)<0: have_neg = True
    else: have_neg=False
    
    if cln_image is None: cln_image = image.copy()
    
    if (thres_a is not None) and (sig_a is not None): 
        assert def_type is None # if thres_a and sig_a are not None, def_type should be None
    elif thres_a is None and sig_a is None:
        if def_type=='emi_img':
            vmax = image.max()
            thres_a = [vmax/1e4, vmax/1e3, vmax/1e2, vmax/1e1]
            sig_a   = [5, 2, 1, 0.5, 0.3]
        elif def_type=='amr_map':
            rlmax = int(cln_image.max() )+1  # refine level max
            rlmin = int(cln_image.min() )    # refine level min
            print(rlmin, rlmax)
            assert rlmin<rlmax-2, 'rlmin larger than rlmax-2, should set thres_a and sig_a manually' 
            thres_a = np.linspace(rlmin, rlmax-1, 30)
            sig_a =   np.linspace(6, 0, 31)
        else: raise Exception('def_type needed')
    else: raise Exception('thres_a and sig_a should be either both None or not None')

    n_subimg = len(sig_a)
    smoothed_image = np.zeros_like( image )
    for i in range(n_subimg):
        
        
        if have_neg == False:
            sub_img = image.copy()
            if i==0:             sub_img[cln_image>thres_a[0] ] = 0
            elif i==n_subimg-1:  sub_img[cln_image<thres_a[-1] ] = 0
            else:                sub_img[ (cln_image<thres_a[i-1] ) | (cln_image>thres_a[i] ) ] = 0
            sub_img_smo = gaussian_filter(sub_img, sig_a[i] )
            
        else:
            sub_pos_img = image.copy()
            sub_neg_img = image.copy()
            if i==0:             
                sub_pos_img[ (cln_image>thres_a[0]) ] = 0
                sub_neg_img[ (cln_image>thres_a[0]) ] = 0
            elif i==n_subimg-1:  
                sub_pos_img[ (cln_image<thres_a[-1]) ] = 0
                sub_neg_img[ (cln_image<thres_a[-1]) ] = 0
            else:                
                sub_pos_img[ (cln_image<thres_a[i-1] ) | (cln_image>thres_a[i] ) ] = 0
                sub_neg_img[ (cln_image<thres_a[i-1] ) | (cln_image>thres_a[i] ) ] = 0
            sub_pos_img[image<0] = 0
            sub_neg_img[image>=0] = 0
                
            sub_pos_img_smo = gaussian_filter(sub_pos_img, sig_a[i] )
            sub_neg_img_smo = -gaussian_filter(-sub_neg_img, sig_a[i] )
            sub_img_smo = sub_pos_img_smo + sub_neg_img_smo
                    
        smoothed_image = smoothed_image + sub_img_smo
        
    return smoothed_image


def adaptive_smoothing_a(image_a, cln_image_a=None, thres_a=None, sig_a=None, def_type=None):
    '''
    Perform adaptive smoothing for an array of images
    '''
    if cln_image_a is None: cln_image_a = image_a.copy()
    
    nimg = len(image_a)
    smo_img_a = np.zeros_like(image_a)
    for i, image, cln_image in zip( range(nimg), image_a, cln_image_a ):
        smo_img = adaptive_smoothing(image, cln_image, thres_a, sig_a, def_type)
        smo_img_a[i] = smo_img 
    return smo_img_a  
        
def bilateralFilter(image, vmin, vmax, log, d=-1, sigmaColor=30, sigmaSpace=10):
    import cv2
    image = 256*num_util.normalize_arr( image, vmin, vmax, log=log )
    image = image.astype('uint8')
    smoothed_image = cv2.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)/256
    smoothed_image = num_util.rev_normalize_arr(smoothed_image, vmin, vmax, log=log)
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