# matplotlib template
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



'''
subplots example
'''

nrow=...
ncol=...
wspace=...
hspace=...
fig, axs = plt.subplots(nrow, ncol, figsize=(5*ncol, 5*nrow) , sharex=True, sharey=True, tight_layout=True, gridspec_kw = {'wspace':wspace, 'hspace':hspace})

axs[0][0]

# set the color of subplots' frame
for spine in axs[0][0].spines.values():
    spine.set_edgecolor('white')

# xy label
axs[0][0].set_xscale('log')
axs[0][0].set_xlabel('...' ,fontsize=18)
axs[0][0].set_xticks([1, 2, 3, 4])
axs[0][0].tick_params(axis="x", labelsize=20)

# subplot_adjusts
bottom=0.05
top=0.97
fontsize=20
plt.subplots_adjust(left=0.06, right=0.89, bottom=bottom, top=top)

# colorbar
cb_frac=0.8
cb_edge_a, lcb = pu.colorbar_edges(nrow, bottom, top, cb_frac, hspace)
cax1 = fig.add_axes([0.90, cb_edge, 0.01, lcb])
cbar1 = fig.colorbar( locals()['p0%d'%(ncol-1)], cax= cax1, ticks=[...]  )
cbar1.ax.tick_params(labelsize=fontsize )
cbar1.set_label('',size=fontsize )


'''
vorbin and plotbin
'''
import vorbin
import plotbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from plotbin.display_bins import *

info = {'nx':image_npix+1, 'ny':image_npix+1}
x_aa, y_aa = coord_util.create_coord(info, size=6, ndim=2, center=True)
x_aa = x_aa.flatten()
y_aa = y_aa.flatten()

signal = image_a[0].flatten()

noise  = np.ones_like(signal)*1e-20
target_sn = 10
pixelsize = 6/200 

bin_number, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(
    x_aa, y_aa, signal, noise, target_sn, cvt=True, pixelsize=pixelsize, plot=True,
    quiet=True, sn_func=None, wvt=True)

plotbin.display_bins.display_bins(x_aa, y_aa, bin_number, np.log10(sn), pixelsize=pixelsize, cmap='magma' )