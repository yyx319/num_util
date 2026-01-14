'''
Functions reading cloudy output
'''

import numpy as np
import matplotlib.pyplot as plt






def read_cloudy_output(filename, Nbins_a, opvar_a):
	'''
	Read the output table from cloudy
	'''
	ndim = len(Nbins_a) # dimension of table


	cloudy_op={}
	if ndim==1:
    	pass
	elif ndim==2:
    	Nbins1, Nbins2 = Nbins_a
    	for var in opvar_a:
        	cloudy_op[var] = np.zeros([Nbins1, Nbins2])


    	with open(filename,'r') as input_file:
        	lines = input_file.readlines()
        	for i in range(Nbins1):
            	for j in range(Nbins2):
                	for k, var in enumerate(opvar_a):
                    	cloudy_op[var][i,j] = float(lines[i*Nbins2+j+1].split()[k+2])
	elif ndim==3:
    	pass
	elif ndim==4:
    	pass


	return cloudy_op




