# basic routine for reading data
import numpy as np
from astropy.io import fits

# fits file
def print_fits_header(hdu):
    '''
    Input:
        hdu: Header Data Unit
    '''
    print('fits header')
    for line in hdu.header.cards:
        print(line)
    print('\n')


