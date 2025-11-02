import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib.colors as colors
import healpy
import h5py
import astropy.units as u; import astropy.constants as c
from scipy import stats
import yt
import pynbody as pn

def cal_G_dl(unit_v, unit_l, unit_m):
    '''
    Calculate dimensionless gravitational constant. 
    '''
    assert unit_v.unit == u.cm/u.s
    assert unit_l.unit == u.cm
    assert unit_m.unit == u.g
    
    unit_t = unit_l/unit_v 
    G_dl = c.G/unit_l**3*unit_m*unit_t**2 
    G_dl = G_dl.decompose().value
    
    return G_dl 

def rescale_G(unit_v, unit_l, unit_m, rescale_factor):
    '''
    Rescale the units to preserve the dimensionless G.
    Suppose u_l -> a u_l, u_m -> b u_m, u_t -> c u_t, we get b c^2 = a^3.
    Input:
        - unit_v, unit_l, unit_m: code units of velocity, length, mass
        - rescale_factor: dictionary specifying the rescale factors for two of a, b, c
    '''
    unit_t = unit_l/unit_v
    if list( rescale_factor.keys() ) == ['b', 'c']:
        b = rescale_factor['b']
        c = rescale_factor['c']
        a = b**(1/3) * c**(2/3)
    elif list( rescale_factor.keys() ) == ['a', 'b']:
        a = rescale_factor['a']
        b = rescale_factor['b']
        c = np.sqrt( a**3/b )
    elif list( rescale_factor.keys() ) == ['a', 'c']:
        a = rescale_factor['a']
        c = rescale_factor['c']
        b = a**3/c**2 
    else:
        raise ValueError('rescale_factor should be a dictionary specifying two of a, b, c')
    
    # rescale    
    rs_unit_l = a*unit_l 
    rs_unit_m = b*unit_m
    rs_unit_t = c*unit_t
    
    rs_unit_v = rs_unit_l/rs_unit_t
    
    return rs_unit_v, rs_unit_l, rs_unit_m 