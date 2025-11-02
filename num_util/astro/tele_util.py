import numpy as np
import astropy.units as u; import astropy.constants as c
from astropy.io import fits
import matplotlib.pyplot as plt

from scipy import interpolate

from num_util.const_util import *
from num_util.path_info import *

########
# JWST #
########

def JWST_dispersion_curve(disperser, lam):
    '''
    Return the dispersion and resolution of JWST NIRSpec disperser for a given wavelength. The curve data is downloaded from: 
    https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters#gsc.tab=0

    Input:
        - disperser: str, the name of the disperser, e.g., 'g140h', 'g235h', 'g395h', 'g140m', 'g235m', 'g395m', 'prism'
        - lam: wavelength, astropy quantity, e.g., 1*u.micrometer
    Output:
        - dispersion: dispersion
        - res: resolution 
    '''

    assert disperser in ['g140h', 'g235h', 'g395h', 'g140m', 'g235m', 'g395m', 'prism']
    
    file = f'{num_util_data_dir}/JWST/jwst_nirspec_{disperser}_disp.fits'
    hdul= fits.open(file)

    #print( hdul.info() )
    #print( hdul[0].header )

    jwst_lam_a = hdul[1].data['WAVELENGTH'] # in micrometer
    jwst_disper_a = hdul[1].data['DLDS']
    jwst_res_a = hdul[1].data['R']

    hdul.close()

    f = interpolate.interp1d(jwst_lam_a, jwst_disper_a)
    dispersion = f( lam.to(u.micrometer).value )

    f = interpolate.interp1d(jwst_lam_a, jwst_res_a)
    res = f( lam.to(u.micrometer).value )
    
    return dispersion, res

