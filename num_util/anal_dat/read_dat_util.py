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

# csv file
def read_csv(filename, delimiter=',', dtype=float):
    '''
    Routine to read csv data
    '''
    import csv
    dat = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for row in csv_reader:
            dat.append(row)
    if dtype in [int, float]:
        dat = np.array(dat, dtype=dtype)
    return dat
