import numpy as np
import astropy.units as u; import astropy.constants as c
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

import num_util 
from num_util.const_util import *




'''
observational calculations relating to image and spectra
https://en.wikipedia.org/wiki/Radiometry
'''


def flam2fnu(flam, lam):
    '''
    Convert intensity from flam to fnu
    '''
    assert lam.unit.is_equivalent(u.Angstrom)
    fnu = flam * lam**2 / c.c
    return fnu 

def fnu2flam(fnu, nu):
    '''
    Convert intensity from fnu to flam
    '''
    assert nu.unit.is_equivalent(u.Hz)
    flam = fnu * nu**2 / c.c
    return flam 

'''
dS is the differential area of galaxy, dA is the differential area of telescope
Luminosity surface density (Lsd) = dE/(dt dS) unit erg/s/cm2
surface brightness (SB) = dE/(dt dA dOmega), unit: erg/s/cm2/arcsec2
'''
def Lsd2SB(Lsd, z):
    '''
    convert luminosity surface density to surface brightness
    '''
    assert Lsd.unit.is_equivalent(u.erg/u.s/u.cm**2)
    SB = Lsd/( 4*np.pi*(1+z)**4*u.sr )
    SB = SB.to(u.erg/u.s/u.cm**2/u.arcsec**2)
    return SB  

def Lpix2SB(Lpix, dx, z):
    '''
    Convert Luminosity in pixel to surface brightness
    '''    
    Lsd = Lpix/dx**2
    SB = Lsd2SB(Lsd, z)
    return SB

def SB2Lsd(SB, z):
    '''
    convert surface brightness to luminosity surface density
    '''
    assert SB.unit.is_equivalent(u.erg/u.s/u.cm**2/u.arcsec**2)
    Lsd=SB*4*np.pi*(1+z)**4*u.sr 
    Lsd = Lsd.to(u.erg/u.s/u.cm**2)
    return Lsd

def SB2Lpix(SB, dx, z):
    Lsd = SB2Lsd(SB, z)
    Lpix = Lsd*dx**2
    Lpix = Lpix.to(u.erg/u.s)
    return Lpix


'''
specific intensity (nu)
'''
def Lppv2spec_int(Lppv, dx, dnu, z):
    '''
    Convert luminosity in ppv voxel to specific intensity
    specific intensity (nu) = dE/(dt dA dnu dOmega )
    unit: erg / s / cm2 / Hz /  arcsec2 
    '''
    L_ppv_d = Lppv/dx**2/dnu
    spec_int = L_ppv_d/( 4*np.pi*(1+z)**4*u.sr )
    spec_int = spec_int.to(u.erg/u.s/u.cm**2/u.Hz/u.arcsec**2)
    return spec_int 

Lppv2spec_int_nu = Lppv2spec_int 

def spec_int_nu2Lppv(spec_int, dx, dnu, z):
    L_ppv_d = spec_int*4*np.pi*(1+z)**4*u.sr
    Lppv = L_ppv_d*dx**2*dnu 
    Lppv = Lppv.to(u.erg/u.s)
    return Lppv

'''
specific intensity (lambda)
'''
def Lppv2spec_int2(Lppv, dx, dl, z):
    '''
    Convert luminosity in ppv voxel to specific intensity
    specific intensity (lambda)  = dE/(dt dA dl dOmega )
    unit: erg / s / cm2 / Ang /  arcsec2 
    '''
    L_ppv_d = Lppv/dx**2/dl
    spec_int = L_ppv_d/( 4*np.pi*(1+z)**4*u.sr )
    spec_int = spec_int.to(u.erg/u.s/u.cm**2/u.Angstrom/u.arcsec**2) # unit conv
    return spec_int 

Lppv2spec_int_l = Lppv2spec_int2

def spec_int_l2Lppv(spec_int, dx, dl, z):
    '''
    Convert specific intensity to luminosity in ppv voxel
    '''
    L_ppv_d = spec_int*4*np.pi*(1+z)**4*u.sr
    Lppv = L_ppv_d*dx**2*dl 
    Lppv = Lppv.to(u.erg/u.s)
    return Lppv 


def cal_AB_mag(spec_flux_nu):
    '''
    Calculate AB magnitude, Oke & Gunn 1983
    '''    
    assert spec_flux_nu.unit.is_equivalent(u.erg/u.s/u.cm**2/u.Hz), f"spec_flux nu unit {spec_flux_nu.unit} is incorrect!"
    spec_flux_nu = spec_flux_nu/(u.erg/u.s/u.cm**2/u.Hz)
    m = -2.5 * np.log10( spec_flux_nu ) - 48.60
    return m
    
def cal_abs_mag(m, DL):
    '''
    Calculate absolute magnitude
    '''
    assert num_util.is_unitless(m) and DL.unit.is_equivalent(u.pc)
    M = m - 5*np.log10( DL/(10*u.pc) ) 
    return M

'''
Spectra
'''
class spectra():
    def __init__(self, x, f, spec_type, verbose=False, **kwargs ):
        '''
        Inputs:
            x: can be wavelength, frequency
            f: represent general intensity quantities, note that most functions only support specific flux and specific intensity
            spec_type: 'emission line', 'absorption line', 'continuum', 'composite emission', 'composite absorption'
        '''
        
        if x.unit.is_equivalent(u.Angstrom):
            self.lam = x
            self.nu = num_util.lam2nu(self.lam)
        elif x.unit.is_equivalent(u.Hz):
            self.nu = x 
            self.lam = num_util.nu2lam(self.nu)
        else:
            raise ValueError('The unit of x is incorrect!')
                
        self.f = f  # general flux quantity
        if type(f)==np.ndarray:
            if 'f_type' in kwargs.keys(): self.f_type = kwargs['f_type']
            else: self.f_type = 'dimensionless' 
        elif f.unit.is_equivalent(u.erg/u.s/u.Hz):                      self.f_type = 'spectral flux (nu)'
        elif f.unit.is_equivalent(u.erg/u.s/u.Angstrom):                self.f_type = 'spectral flux (lam)'
        elif f.unit.is_equivalent(u.erg/u.cm**2/u.s/u.Hz):              self.f_type = 'specific flux (nu)'
        elif f.unit.is_equivalent(u.erg/u.cm**2/u.s/u.Angstrom):        self.f_type = 'specific flux (lambda)'
        elif f.unit.is_equivalent(u.erg/u.cm**2/u.s/u.Hz/u.sr):         self.f_type = 'specific intensity (nu)'
        elif f.unit.is_equivalent( u.erg/u.cm**2/u.s/u.Angstrom/u.sr ): self.f_type = 'specific intensity (lambda)'
        else:raise ValueError('The unit of f is incorrect!')
            
        if '(nu)' in self.f_type:
            self.f_nu = f 
            self.f_lam = fnu2flam(f, self.nu)
        elif '(lambda)' in self.f_type:
            self.f_lam = f
            self.f_nu = flam2fnu(f, self.lam)
            
        self.spec_type=spec_type 
        
        self.verbose=verbose
        
    def cal_vsys(self):
        '''
        Calculate the systemic velocity
        '''
        lam, f = self.lam, self.f
        self.lam_sys = np.trapz(lam*f, lam)/np.trapz(f, lam)
    
    def telescope(self, lam_line, R):
        '''
        Modelling telescope effect
        R = lambda/delta lambda
        '''
        lam = self.lam
        sigma_l = lam_line/R/2.355
        sigma = sigma_l * len(lam ) / ( np.max(lam) - np.min(lam) )
        sigma = sigma.decompose().value
        self.f_conv =  gaussian_filter( self.f, sigma=sigma )
        if self.f_type !='dimensionless':
            self.f_conv *= self.f.unit 
            self.f_nu_conv =  gaussian_filter( self.f_nu, sigma=sigma )*self.f_nu.unit 
            self.f_lam_conv = gaussian_filter( self.f_lam, sigma=sigma )*self.f_lam.unit 
                    
        dlam = self.lam[1]-self.lam[0]
        res = sigma_l # telescope resolution
        fac = int( res/dlam ) # resolution ratio (size of res in dlam)
        
        N_throw = len(lam)%fac    # size that gets thrown away during rebinning
        N = len(lam) - N_throw    # size before rebinning
        N_rb = int( N/fac ) # size after rebinning
        
        if self.verbose: print(f'dlam={dlam}, res={res}, arr_size={len(lam)}, N_throw={N_throw}, N_rb={N_rb}')

        if N_throw !=0:
            self.lam_tel = np.append( num_util.rebin(self.lam[:N], N_rb),    np.mean(self.lam[N:]) )
            self.nu_tel =  np.append( num_util.rebin(self.nu[:N], N_rb),    np.mean(self.nu[N:]) )
            self.f_tel =   np.append( num_util.rebin(self.f_conv[:N], N_rb), np.mean(self.f_conv[N:]) )
            if self.f_type !='dimensionless':
                self.f_nu_tel =   np.append( num_util.rebin(self.f_nu_conv[:N], N_rb), np.mean(self.f_nu_conv[N:]) )
                self.f_lam_tel =   np.append( num_util.rebin(self.f_lam_conv[:N], N_rb), np.mean(self.f_lam_conv[N:]) )
        else:
            self.lam_tel =  num_util.rebin(self.lam, N_rb)       
            self.nu_tel =   num_util.rebin(self.nu, N_rb)       
            self.f_tel =    num_util.rebin(self.f_conv, N_rb) 
            if self.f_type !='dimensionless':
                self.f_nu_tel =    num_util.rebin(self.f_nu_conv, N_rb) 
                self.f_lam_tel =    num_util.rebin(self.f_lam_conv, N_rb) 
            
    def cal_mean_flux(self, band, type='normal'):
        '''
        Calculate mean flux in the band
        '''
        if type in ['normal', 'conv']:
            lam = self.lam
        elif type=='telescope':
            lam = self.lam_tel
        idx = np.where( (lam>band[0]) & (lam<band[1]) )[0]
        
        if type=='normal':
            mean_f = np.mean( self.f[idx] )
            return mean_f
        elif type=='conv':
            if self.f_type !='dimensionless':
                mean_f_nu = np.mean( self.f_nu_conv[idx] )
                mean_f_lam = np.mean( self.f_lam_conv[idx] )
                return mean_f_nu, mean_f_lam
            else:
                mean_f = np.mean( self.f_conv[idx] )
                return mean_f
        elif type=='telescope':
            if self.f_type !='dimensionless':
                mean_f_nu = np.mean( self.f_nu_tel[idx] )
                mean_f_lam = np.mean( self.f_lam_tel[idx] )
                return mean_f_nu, mean_f_lam
            else:
                mean_f = np.mean( self.f_tel[idx] )
                return mean_f
        else: raise ValueError('type is incorrect!')
        

    def cal_total_luminosity(self, x, z, f_frame, lam_frame, type='normal',  ):
        '''
        calculate total luminosity
        '''
        assert self.f_type in ['specific flux (nu)', 'specific intensity (nu)', 'specific flux (lambda)', 'specific intensity (lambda)'], f'f_type {self.f_type} is incorrect!'
        if f_frame=='rest': 
            if type=='normal': self.f_rest = self.f
            elif type=='conv': self.f_conv_rest = self.f_conv
        elif f_frame=='obs': 
            if '(lambda)' in self.f_type: 
                if type=='normal': self.f_rest = self.f*(1+z)
                elif type=='conv': self.f_conv_rest = self.f_conv*(1+z)
            elif '(nu)' in self.f_type:
                if type=='normal': self.f_rest = self.f/(1+z)
                elif type=='conv': self.f_conv_rest = self.f_conv/(1+z)
        else: raise ValueError('f_frame is incorrect!')
        
        if lam_frame=='rest': 
            self.lam_rest = self.lam
            self.nu_rest = self.nu
        elif lam_frame=='obs': 
            self.lam_rest = self.lam/(1+z)
            self.nu_rest = self.nu*(1+z)
        else: raise ValueError('lam_frame is incorrect!')
            
        DA = cosmo.angular_diameter_distance(z=z)
        omega = x**2/(DA**2)*u.sr
        if type == 'normal':
            if 'specific flux' in self.f_type:        f = self.f_rest / omega
            elif 'specific intensity' in self.f_type: f = self.f_rest
        elif type == 'conv':
            if 'specific flux' in self.f_type:        f = self.f_conv_rest / omega  # convert to specific intensity
            elif 'specific intensity' in self.f_type: f = self.f_conv_rest
            
        if '(lambda)' in self.f_type:
            SB = np.trapz(f, self.lam_rest )
        elif '(nu)' in self.f_type:
            SB = np.trapz(f, self.nu_rest)
            
        elif type== 'telescope':
            if self.f_type == 'specific intensity (lambda)':
                SB = np.trapz(self.f_tel, self.lam_tel )
            elif self.f_type=='specific intensity (nu)':
                SB = np.trapz(self.f_tel, self.nu_tel)      
            else: raise ValueError(f'f_type {self.f_type} is incorrect!')  
        L = SB2Lpix(SB, x, z)    
        return L
    
    def cal_UV_beta(self):
        '''
        Calculate UV continuum slope beta, Calzetti+1994
        TBW
        '''
        lam_uvc_min, lam_uvc_max = 1250*u.Angstrom, 2600*u.Angstrom
        beta = 0
        return beta 

        
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
        self.r2d, self.phi2d = num_util.polar_coord(self.x2d, self.y2d, center=[0,0]) 
        
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
        image = num_util.rebin(image, npix_tele, npix_tele)  # resolution
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
        ppv_smo = num_util.rebin(self.ppv, npix_tele, npix_tele, nspec_tele )  # resolution
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
    
def cal_intr_img_gas(x,y,z, lum, LOS, image_npix, size, redshift):

    dx = size/image_npix
    x_bin = np.linspace(-size/2, size/2, image_npix+1)
    wx, wy = num_util.cal_projection_pos(x, y, z, LOS )
    _,_,img =  num_util.phase_diagram(wx, wy, lum.value, x_bin, x_bin )
    img = Lpix2SB(img*u.erg/u.s, dx, redshift)
    
    return img 

def cal_intr_spec_gas(v_vec, sig, lum, LOS, nu_a, nu_0, size, z):
    '''
    Inputs:
        v_vec:
        sig: doppler width of gas cell
        lum:
        
        LOS:
        nu_a: 
    '''
    lam_a = num_util.nu2lam(nu_a)
    dl = lam_a[1]-lam_a[0]
    ngas = len(lum)
    dnu = np.abs( nu_a[1]-nu_a[0] )
    
    # doppler factor x, (ngas, 1)
    x = 1 + np.dot( v_vec, LOS  )/c.c
    x = x.reshape( ngas, 1 )
    
    nu_2d, sig_2d = np.meshgrid( nu_a, sig ) 
    spec_cell = lum.reshape( ngas, 1 )/x*num_util.nor_gaussian( nu_2d/x, nu_0, sig_2d)*dnu # 2d (nspec, ngas)
    print(spec_cell.unit)
    
    intr_spec = np.sum( spec_cell, axis=0 ) # dim: nspec; 
    intr_spec = Lppv2spec_int2(intr_spec, size, dl, z)
    return intr_spec

