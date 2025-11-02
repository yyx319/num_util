import numpy as np
from astropy.modeling.models import Sersic1D
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

from num_util.const_util import *


class halo():
    '''
    halo class
    '''
    def __init__(self, pos, vel, mass, rvir):
        self.pos=pos
        self.vel=vel
        self.mass=mass
        self.rvir=rvir

class BH():
    '''
    black hole class
    '''
    def __init__(self, pos, vel, mass):
        self.pos = pos
        self.vel = vel
        self.mass = mass 
        
class galaxy():
    '''
    galaxy class
    '''
    def __init__(self, pos_gal, vel_gal, mass_gal, pos_gas, vol_gas, dens_gas, v_gas):
        self.pos_gal = pos_gal
        self.vel_gal = vel_gal
        self.mass_gal = mass_gal 

        # gas 
        self.pos_gas = pos_gas 
        self.vol_gas = vol_gas
        self.dens_gas = dens_gas
        self.v_gas = v_gas 
        
        self.x_gas, self.y_gas, self.z_gas = self.pos_gas
        self.vx_gas, self.vy_gas, self.vz_gas = self.v_gas
        self.mass_gas = self.dens_gas*self.vol_gas 

        self.vcx_gas = np.nansum( self.mass_gas*self.vx_gas ) / np.nansum( self.mass_gas )
        self.vcy_gas = np.nansum( self.mass_gas*self.vy_gas ) / np.nansum( self.mass_gas )
        self.vcz_gas = np.nansum( self.mass_gas*self.vz_gas ) / np.nansum( self.mass_gas )
        self.vc_gas = np.array( [self.vcx_gas, self.vcy_gas, self.vcz_gas] )

        # star 

        # DM

        # BH
        
###############
# merger util #
###############

def cal_merger_basic(system_a ):
    '''
    galaxy mergers    
    '''
    nsys = len(system_a)
    
    pos_a = np.zeros( (nsys, 3) )
    vel_a =  np.zeros( (nsys, 3) )
    mass_a = np.zeros( nsys )
    for i in range(nsys):
        pos_a[i]  = system_a[i].pos 
        vel_a[i]  = system_a[i].vel
        mass_a[i] = system_a[i].mass
        
    mg_cen = np.sum( pos_a, axis=0 )/nsys
    mg_com = np.sum( pos_a * mass_a.reshape( nsys,1 ), axis=0 )/np.sum( mass_a )
    mg_vel = np.sum( vel_a * mass_a.reshape( nsys,1 ), axis=0 )/np.sum( mass_a )
    
    return mg_cen, mg_com, mg_vel


