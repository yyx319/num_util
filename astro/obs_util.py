import numpy as np
import astropy.units as u; import astropy.constants as c
from astropy.modeling.models import Sersic1D
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

import coord_util
from const_util import *

import vorbin
import plotbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from plotbin.display_bins import *


'''
observational calculations
'''

def Lsd2SB(Lsd, z):
    '''
    convert luminosity surface density to surface brightness
    dS is the differential area of galaxy, dA is the differential area of telescope
    Luminosity surface density (Lsd) = dE/(dt dS) unit erg/s/cm2
    surface brightness (SB) = dE/(dt dA dOmega), unit: erg/s/cm2/arcsec2
    '''
    SB = Lsd/( 4*np.pi*(1+z)**4*u.sr )
    SB = SB.to(u.erg/u.s/u.cm**2/u.arcsec**2)
    return SB  

def SB2Lsd(SB, z):
    '''
    convert surface brightness to luminosity surface density
    '''
    Lsd=SB*4*np.pi*(1+z)**4*u.sr 
    Lsd = Lsd.to(u.erg/u.s/u.cm**2)
    return Lsd

def Lpix2SB(Lpix, dx, z):
    '''
    Convert Luminosity in pixel to surface brightness
    '''    
    Lsd = Lpix/dx**2
    SB = Lsd2SB(Lsd, z)
    return SB

def Lppv2spec_int(Lppv, dx, dnu, z):
    '''
    Convert luminosity in ppv voxel to specific intensity
    specific intensity = dE/(dt dA dnu dOmega )
    unit: erg / s / cm2 / Hz /  arcsec2 
    '''
    L_ppv_d = Lppv/dx**2/dnu
    spec_int = L_ppv_d/( 4*np.pi*(1+z)**4*u.sr )
    spec_int = spec_int.to(u.erg/u.s/u.cm**2/u.Hz/u.arcsec**2)
    return spec_int 

def Lppv2spec_int2(Lppv, dx, dl, z):
    '''
    Convert luminosity in ppv voxel to specific intensity 2
    specific intensity 2  = dE/(dt dA dl dOmega )
    unit: erg / s / cm2 / Ang /  arcsec2 
    '''
    L_ppv_d = Lppv/dx**2/dl
    spec_int = L_ppv_d/( 4*np.pi*(1+z)**4*u.sr )
    spec_int = spec_int.to(u.erg/u.s/u.cm**2/u.Angstrom/u.arcsec**2)
    return spec_int 

'''
Spectra
'''
class spectra():
    def __init__(self, x, f ):
        '''
        x can be wavelength, frequency or velocity
        '''
        if x.unit.is_equivalent(u.Angstrom):
            self.lam = x
        elif x.unit.is_equivalent(u.Hz):
            self.nu = x 
                
        if f.unit.is_equivalent(u.erg/u.s/u.Hz) or f.unit.is_equivalent(u.erg/u.s/u.Angstrom):
            self.dir_spec_int = f 
        elif f.unit.is_equivalent(u.erg/u.cm**2/u.s/u.Hz) or f.unit.is_equivalent(u.erg/u.cm**2/u.s/u.Angstrom):
            self.spec_flux = f
            self.SB = self.spec_int 
        elif f.unit.is_equivalent(u.erg/u.cm**2/u.s/u.Hz/u.sr) or f.unit.is_equivalent( u.erg/u.cm**2/u.s/u.Angstrom/u.sr):
            self.spec_int = f
        else:
            raise ValueError('The unit of f is incorrect!')
            
        
    def cal_vsys(self):
        pass 
    
    def cal_total_luminosity(self):
        pass
        

class Lya_spectra(spectra):
    def __init__(self, x, flux):
        pass 
    
    
'''
image analysis
'''
class map():
    pass 

class image():
    '''
    image class, 
    default adopt cartesian grid, with centre (0,0)
    '''
    def __init__(self, x, y, SB):
        self.x = x # 1D
        self.y = y # 1D
        self.SB = SB # 2D
        
        # derived quantities
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        self.x2d, self.y2d = np.meshgrid( self.x, self.y, indexing='ij')
        self.r2d, self.phi2d = coord_util.polar_coord(self.x2d, self.y2d, center=[0,0]) 
        
    def cal_scale_radius(self, method):
        r_sca = np.sum(self.r2d*self.SB ) / np.sum( self.SB )
        return r_sca

    def Centroid(self):
        cen_x = np.sum(self.SB*self.x2d)/np.sum(self.SB)
        cen_y = np.sum(self.SB*self.y2d)/np.sum(self.SB)
        return cen_x, cen_y
        
    def PAandAxisratio(self):
        cen_x, cen_y = self.Centroid()
        X2 = np.sum(self.SB*self.x2d**2)/np.sum(self.SB) - cen_x**2
        Y2 = np.sum(self.SB*self.y2d**2)/np.sum(self.SB) - cen_y**2
        XY = np.sum(self.SB*self.x2d*self.y2d )/np.sum(self.SB) - cen_x*cen_y
    
        # calculating axis, angle
        term1 = (X2+Y2)/2.
        term2 = np.sqrt( ( (X2-Y2)/2 )**2 + XY**2 )
        a = 2*np.sqrt( term1 + term2 )
        b = 2*np.sqrt( term1 - term2 )
        theta0 = 1/2 * np.arctan( 2*XY/(X2-Y2) )
        return a, b, theta0

    def make_tele_image(self, noise, npix_tele, beam_size):
        '''
        add noise and beam smearing effect
        '''
        image  = gaussian_filter(self.SB, sigma=beam_size )      # beam smearing
        image = coord_util.rebin(image, npix_tele, npix_tele)  # resolution
        image = np.random.normal(image, noise) # add noise
        image[image<3*noise] = 0 # mask the noise
        return image
    
    def rotate_2d(self, center, angle, n_g):
        '''
        rotate the spatial part of the ppv cube and zoom in
        note that n_g should be even    
        '''
        (x0, y0) = center
        pp_rot = np.zeros((n_g, n_g))
        nx, ny = np.shape(self.SB )
        
        if angle == 0:
            pp_rot = self.SB[int(x0-n_g/2):int(x0+n_g/2),
                        int(y0-n_g/2):int(y0+n_g/2)]
        elif angle != 0:
            c = np.cos(angle*np.pi/180.)
            s = np.sin(angle*np.pi/180.)
            for x in range(n_g):
                for y in range(n_g):
                    src_x = int( c*(x-n_g/2) + s*(y-n_g/2) + x0 )
                    src_y = int( -s*(x-n_g/2) + c*(y-n_g/2) + y0 )
                    if 0<=src_x<nx and 0<src_y<ny:
                        pp_rot[x, y] = self.SB[int(src_x), int(src_y)]
                    else:
                        pp_rot[x, y] = -1 # we do not set it to zero to avoid numerical uncertainty
        return pp_rot

    def isophote(self, Lum_a):
        # TBW
        r_a = [] 
        for Lum in Lum_a:
            idx = np.where( self.SB>Lum )[0]
            size = len(idx)* self.dx*self.dy
            r_eff = np.sqrt( size/np.pi ) 
            r_a.append(r_eff)
        return r_a


'''
PPV cube analysis
'''
class PPV_cube():
    def __init__(self, x,y,v, ppv, quantity):
        self.x = x
        self.y = y
        self.v = v 
        self.ppv = ppv 
        self.quantity = quantity # physical quantity 

        self.nx = len(self.x)
        self.ny = len(self.y)
        self.nv = len(self.v) 

    def make_tele_ppv(self, noise, nspec_tele, npix_tele, beam_size):
        # ppv (npix, npix, nspec)
        for i in self.nv:
            self.ppv[:,:,i] = gaussian_filter(self.ppv[:,:,i], sigma=beam_size)       # beam smearing
        ppv_smo = coord_util.rebin(self.ppv, npix_tele, npix_tele, nspec_tele )  # resolution
        ppv_tele = np.random.normal(ppv_smo, noise) # add noise
        ppv_tele[ppv_tele<3*noise]=0 # mask the noise 
        return ppv_tele
    
    def rotate(self, center, angle, n_g):
        # rotate the spatial part of the ppv cube and zoom in
        # note that n_g should be even
        (x0, y0) = center
        n_v = np.shape(self.ppv)[0]
        ppv_rot = np.zeros((n_v, n_g, n_g))
        if angle == 0:
            ppv_rot = self.ppv[:, int(x0-n_g/2):int(x0+n_g/2),
                        int(y0-n_g/2):int(y0+n_g/2)]
        elif angle != 0:
            c = np.cos(angle*np.pi/180.)
            s = np.sin(angle*np.pi/180.)
            for x in range(n_g):
                for y in range(n_g):
                    src_x = c*(x-n_g/2) + s*(y-n_g/2) + x0
                    src_y = -s*(x-n_g/2) + c*(y-n_g/2) + y0
                    ppv_rot[:, x, y] = self.ppv[:, int(src_x), int(src_y)]

        return ppv_rot 
    
    def cal_total(self):
        ''' 
        calculate the total of quantity
        '''
        pass
    
    def vel_dist(self):
        '''
        calculate the velocity distribution of the quantity
        '''
        pass 
    
    def zeroth_moment_map(self, noise):
        '''
        calculate the zeroth moment
        '''
        dv = (self.v[-1]-self.v[0])/(len(self.v))
        L_int = np.sum(self.ppv*dv, axis=0)
        noise_int = noise*dv*np.sqrt(len(self.v))
        good_pix_x_a, good_pix_y_a = np.where(L_int > 3*noise_int)
        self.ppv[self.ppv < 3*noise] = 0

        # calculate second moment in 'good' pixels
        n_v, n_pix_x, n_pix_y = np.shape(self.ppv)
        m0 = np.zeros((n_pix_x, n_pix_y))
        for x, y in zip(good_pix_x_a, good_pix_y_a):
            lum_los = self.ppv[:, x, y]  # lum array in beam
            m0[x, y] = np.sum(lum_los)*dv
        return m0

    def first_moment_map(self, noise):
        # output: zeroth moment map m0; first moment map m1
        m0 = self.zeroth_moment_map(noise)

        dv = (self.v[-1]-self.v[0])/(len(self.v))
        L_int = np.sum( self.ppv*dv, axis=0 )
        noise_int = noise*dv*np.sqrt(len(self.v))
        good_pix_x_a, good_pix_y_a = np.where(L_int > 3*noise_int)
        self.ppv[self.ppv < 3*noise] = 0

        # calculate second moment in 'good' pixels
        n_v, n_pix_x, n_pix_y = np.shape(self.ppv)

        m1 = np.zeros((n_pix_x, n_pix_y))
        for x, y in zip(good_pix_x_a, good_pix_y_a):
            lum_los = self.ppv[:, x, y]  # lum array in beam
            m1[x, y] = np.sum(self.v * lum_los*dv)/m0[x, y]
        return m0, m1

    def second_moment_map(self, noise):
        # output: zeroth moment map m0; first moment map m1; second moment map m2
        m0, m1 = self.first_moment_map(noise)

        dv = (self.v[-1]-self.v[0])/(len(self.v))
        L_int = np.sum( self.ppv*dv, axis=0)
        noise_int = noise*dv*np.sqrt(len(self.v))
        good_pix_x_a, good_pix_y_a = np.where(L_int > 3*noise_int)
        self.ppv[self.ppv < 3*noise] = 0

        # calculate second moment in 'good' pixels
        n_v, n_pix_x, n_pix_y = np.shape(self.ppv)

        m2 = np.zeros((n_pix_x, n_pix_y))
        for x, y in zip(good_pix_x_a, good_pix_y_a):
            lum_los = self.ppv[:, x, y]               # lum array in beam
            m2[x, y] = np.sum((self.v-m1[x, y])**2*lum_los*dv)/m0[x, y]
        m2 = np.sqrt(m2)
        return m0, m1, m2
    

    
class emi_PPV_cube(PPV_cube):
    pass 
    
    