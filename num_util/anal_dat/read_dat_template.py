from astropy.io import fits
import num_util

with fits.open('data/Jones_NOV11.fits') as hdul:
    hdul.info() # Print summary of the file
    
    primary_hdu = hdul[0]
    num_util.print_fits_header(primary_hdu)
    primary_data = primary_hdu.data  # Access the data

    data_hdu = hdul[1]
    data = data_hdu.data
    print("Column names:", data.columns)
    field_l = ['zLyaR1000', 'fesclyaHBNoDCR1000', 'dfesclyaHBNoDCR1000', 'REWLyaR1000', 'dREWLyaR1000','ILyaR1000', 'dILyaR1000', 'dvLyaR1000P', 'ddvLyaR1000P' ]
    for field in field_l:
        locals()[f'{field}_J24'] = data[field]