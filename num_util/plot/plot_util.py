import os
import sys
from cProfile import label
from lib2to3.pygram import python_grammar_no_print_statement
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import yt

import astropy.units as u; import astropy.constants as c
from scipy import stats
sys.path.append('/home/yy503/Desktop/simulation_code/ramses_tools/MakeMovies')
from support_functions import *

from astropy.cosmology import WMAP7
import num_util
from num_util.const_util import *


###################
# polish the plot #
###################
def sci_fm(x, prec=2, multi='times', inc_dol=True, verbose=True):
    '''
    change x to scientific format (for publication)
    '''
    if type(x) is str:
        x = float(x)
    
    assert isinstance(prec, int) and (prec>=0), 'prec must be a non-negative integer!'

    a,b = eval(f" '{x:.{prec}e}'.split('e') ")
    
    # formating exponent
    if len(b)==3 and b[:2]=='+0': 
        b = b[2]
    elif len(b)==3 and b[:2]=='-0':
        b = b[0]+b[2]
    else:
        pass 
    
    if inc_dol:
        if b=='0':
            x_sci_fm = f'${a}$'
        else:
            x_sci_fm = f'${a}\\{multi} 10^{{{b}}}$' 
    else:
        if b=='0':
            x_sci_fm = f'{a}'
        else:
            x_sci_fm = f'{a}\\{multi} 10^{{{b}}}'      
        
    if verbose:
        print(f'x={x}, x_sci_fm={x_sci_fm}')
    return x_sci_fm 


def set_legend_handle_size(legend, s):
    for handle in legend.legendHandles:
        handle._sizes = [s]  # Adjust marker size in legend


##################
# Plot templates #
##################

def scatter_hist(x1, y1, ax, ax_histx, ax_histy, bins_x, bins_y, plot='hist'):
    #https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    if plot=='hist':
        ax.hist2d(x1, y1, bins=[bins_x, bins_y])
    elif plot=='scatter':
        ax.scatter(x1, y1, s=3)

    # now determine nice limits by hand:
    ax_histx.hist(x1, bins=bins_x, histtype='step')
    ax_histy.hist(y1, bins=bins_y, orientation='horizontal', histtype='step')

def set_twin(ax, twin_xlabel, xl, xr, x_a, xtwin_a):
    '''
    Set twin x axis
    '''
    twin = ax.twiny()
    twin.set_xlabel(twin_xlabel)
    twin.set_xlim( xl, xr )
    
    twin.set_xticks(x_a)
    twin.set_xticklabels(xtwin_a )

##############################
# function for color of plot # 
##############################

def gen_color_arr(n, cmap='Blues'):
    '''
    generate array representing different depth of with the same color
    '''
    norm = colors.Normalize(vmin=-0.5, vmax=1)
    cmap1 = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap( cmap ) )
    rg = range(n)
    rgb_a = []
    for i in rg:
        rg2 = float( i-min(rg) )/(len(rg)-1) 
        rgb_a.append( cmap1.to_rgba(rg2) )
    return rgb_a

generate_color_array = gen_color_arr 

def generate_color_array_2(x_a, norm=None, cmap='Blues'):
    '''
    generate color array from x_a
    '''
    if norm is None:
        norm = colors.Normalize(vmin=np.min(x_a), vmax=np.max(x_a))
    cmap1 = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap( cmap ) )
    rgb_a = []
    for x in x_a:
        rgb_a.append( cmap1.to_rgba(x) )
    return rgb_a

def gen_misc_color_arr(n, method='default'):
    '''
    Generate color array with different color.
    '''
    if method=='default':
        color_a = plt.rcParams['axes.prop_cycle'].by_key()['color'][:n]
    else:
        raise Exception(f'method {method} not included!')
    return color_a 
    
generate_misc_color_array = gen_misc_color_arr

'''
Construct the frame of the plot
'''
def cal_fig_size(nrow, ncol, wspace, hspace, boundary, panel_ratio=1, mode='publication'):
    '''
    Calculate the figure size
    '''
    if type(boundary) in [list, tuple, np.ndarray]:
        bottom, top, left, right = boundary
    
    if panel_ratio=='for_healpix':
        panel_ratio=0.5
    elif panel_ratio=='for_image':
        panel_ratio=1

    # set figure width
    if mode in ['pub', 'publication']:
        fig_w = 25 
    elif mode in ['pub_single_col', 'publication_single_column']:
        fig_w = 12.5
    elif mode=='normal':
        fig_w = ncol*6
    elif mode=='factor':
        fig_w = 1 
    else:
        raise Exception('mode not recognized!')
    
    panel_w = fig_w * ( right - left ) / (ncol + (ncol-1)*wspace )  # panel width in inches
    panel_h = panel_w*panel_ratio # panel height in inches
    fig_h = (nrow + (nrow-1)*hspace ) * panel_h / ( top-bottom )  # in inches
    
    return fig_w, fig_h

calculate_fig_size = cal_fig_size

def colorbar_edges(nrow, bottom, top, cb_frac, hspace, orientation='vertical'):
    '''
    generate the lower edges of color bars from top to bottom 
    nrow: number of rows 
    bottom, top: bottom and top of the figure
    cb_frac: fraction of colorbar height 
    hspace: same parameters in plt.subplots()
    '''
    panel_h = (top-bottom) / (nrow + (nrow-1)*hspace ) 
    if orientation=='vertical':
        lcb = panel_h*cb_frac  # length of colorbar
        d = panel_h*(1-cb_frac)/2 # vertical distance between colorbar bottom (top) edge to panel bottom (top) edge
        cb_edge_a = np.linspace(top-panel_h+d, bottom+d, nrow) # colorbar lower edges
        return cb_edge_a, lcb
    
    elif orientation=='horizontal':
        wcb = panel_h*hspace*cb_frac 
        d = panel_h*hspace*(1-cb_frac)
        d1 = d*0.1
        d2 = d*0.9
        cb_edge_a = np.linspace( top-panel_h-d1-wcb, bottom-d1-wcb, nrow)
        return cb_edge_a, wcb

colorbar_edges_t2b = colorbar_edges

def colorbar_edges_l2r():
    pass 

def colorbar_edges_all_panel():
    pass


##############
# basic plot # 
##############
def general_1d_hist(var_a, var_bin, field_a, weight_field_a=None, stat='mean'):
    '''
    This is the general histogram
    If field_a=None, weight_field_a=None, and stat='sum', then return the normal histogram of var.
    If field_a is not None, weight_field_a=None, then plot field versus var. 
    If field_a is not None, weight_field_a is not None, then plot field weighted by weight_field versus var. 
    Some example usage:
        1. radial profile: general_1d_hist(r_a, r_bin, field_a, weight_field_a, stat='mean')
    '''
    assert var_a is not None, field_a is not None
    
    if var_a.ndim>=2:
        var_a = var_a.flatten()
    if field_a.ndim>=2:
        field_a = field_a.flatten()
    if weight_field_a is not None and weight_field_a.ndim>2:
        weight_field_a = weight_field_a.flatten()
        
    if weight_field_a is None:
        f_var, _, _ = stats.binned_statistic(var_a, field_a, stat, var_bin)
    else:
        fwf_r, _, _ = stats.binned_statistic(var_a, field_a*weight_field_a, stat, var_bin)
        wf_r, _, _ = stats.binned_statistic(var_a, weight_field_a, stat, var_bin)
        f_var = fwf_r/wf_r 
    var_a2 = 1/2*( var_bin[1:] + var_bin[:-1] )
    return var_a2, f_var  

radial_profile = general_1d_hist 

    
    
def PDF2CDF(x, PDF, type='larger'):
    '''
    Convert 'PDF' to CDF, 'PDF' doesn't need to be normalized.
    type 'larger' means P(>x), 'lower' means P(<x)
    '''
    PDF = PDF/np.trapz(PDF, x) # normalize PDF
    x_bin = num_util.bin_from_array(x)
    dx = x_bin[1:]-x_bin[:-1]
    pxdx = PDF*dx 
    if type=='larger':
        CDF = np.cumsum( pxdx[::-1] )[::-1]
    elif type=='lower':
        CDF = np.cumsum( pxdx )
    else:
        raise Exception('type not recognized!')
    return CDF

def general_2d_histogram(cube_field1, cube_field2, cube_field3=None, f1_bin=None, f2_bin=None, cube_weight_field=None, stat='sum'):
    '''
    x y coordinate field 1, 2; value taken as field 3, weighted field is optional
    '''
    if cube_field3 is None:
        cube_field3 = np.ones_like(cube_field1)
        
    if cube_field1.ndim>=2:
        cube_field1 = cube_field1.flatten()
        cube_field2 = cube_field2.flatten()
        cube_field3 = cube_field3.flatten()
    
    if f1_bin is None:
        f1_bin = np.linspace( np.min(cube_field1), np.max(cube_field1), 100 )
    if f2_bin is None:
        f2_bin = np.linspace( np.min(cube_field2), np.max(cube_field2), 100 )
        

    
    if cube_weight_field is None:
        ret = stats.binned_statistic_2d(cube_field1, cube_field2, cube_field3, statistic=stat, bins=[f1_bin, f2_bin] )
        f3_aa = ret.statistic
    else:
        if cube_weight_field.ndim>2:
            cube_weight_field = cube_weight_field.flatten()
        ret1 = stats.binned_statistic_2d(cube_field1, cube_field2, cube_field3*cube_weight_field, statistic=stat, bins=[f1_bin, f2_bin] )
        ret2 = stats.binned_statistic_2d(cube_field1, cube_field2, cube_weight_field, statistic=stat, bins=[f1_bin, f2_bin] )
        f3_aa = ret1.statistic/ret2.statistic      

    f1_a = 1/2*(f1_bin[1:]+f1_bin[:-1])
    f2_a = 1/2*(f2_bin[1:]+f2_bin[:-1])
    f3_aa = f3_aa.transpose() # used for plt.imshow
    return f1_a, f2_a, f3_aa

phase_diagram = general_2d_histogram


def get_relation(x_a, y_a, x_bin=None, stat='median', logx=False):
    if x_a.ndim>=2:
        x_a = x_a.flatten()
    if y_a.ndim>=2:
        y_a = y_a.flatten()

    if x_bin is None:
        if logx:
            x_bin = np.linspace( np.min(x_a), np.max(x_a), 10 )
        else:
            logx_min = np.log10( np.min(x_a) )
            logx_max = np.log10( np.max(x_a) )
            x_bin = np.logspace( logx_min, logx_max, 10 ) 

    y_a2, _, _ = stats.binned_statistic(x_a, y_a, stat, x_bin)

    if logx:
        x_a2 = np.sqrt( x_bin[1:]*x_bin[:-1] )
    else:
        x_a2 = 1/2*(x_bin[1:]+x_bin[:-1])        
    return x_a2, y_a2

def cal_rep_error_bar_log(x_a, xerr_m_a, xerr_p_a, xrep):
    # remove 0 values
    idx = np.where( x_a!=0 )[0]
    x_a, xerr_m_a, xerr_p_a = x_a[idx], xerr_m_a[idx], xerr_p_a[idx]

    logx_a = np.log10(x_a)
    logx_err_m_a = logx_a - np.log10(x_a - xerr_m_a)
    logx_err_p_a = np.log10(x_a + xerr_p_a) - logx_a
    logx_rep_err_m = np.mean(logx_err_m_a)
    logx_rep_err_p = np.mean(logx_err_p_a)

    logxrep = np.log10(xrep)
    xrep_err_m = [xrep - 10**( logxrep-logx_rep_err_m ) ]
    xrep_err_p = [10**( logxrep+logx_rep_err_p ) - xrep ]
    return xrep_err_m, xrep_err_p

#########
# routine for calculating SFR, inflow and outflow
#####################################################








######
# yt #
######
def make_yt_image(ds, normal, fields, weight_field, center, width, another_code='RASCAS', plot_type='prj' ):
    '''
    Make yt image. Also make orientation consistent with the image made by other codes (RASCAS). 
    yt orientation rule: if normal is not xhat, yhat, zhat, then normal is treated as z axis, north_vector is treated as y axis. The coordiate is right-handed and x axis is determined. 
    Check yt.visualization.volume_rendering.off_axis_projection.py for details.
    '''
    buff_size = (200,200)

    if another_code=='RASCAS':
        # special cases: along x axis, normal = (pm1, eps, eps) 
        if np.abs(normal[0])==1:
            # rascas y,z; yt y,z 
            if plot_type in ['prj', 'projection']:
                p = yt.ProjectionPlot(ds, normal='x', fields=fields,  weight_field=weight_field, center=center, width = width, buff_size=buff_size )
            elif plot_type in ['slc', 'slice']:
                p = yt.SlicePlot(ds, normal='x', fields=fields, center=center, width = width, buff_size=buff_size )
            img=p.to_fits_data()[0].data.value
        # special cases: along y or z axis
        elif np.array_equal(normal, [0,1,0]) or np.array_equal(normal, [0,-1,0]) or np.array_equal(normal, [0,0,1]) or np.array_equal(normal, [0,0,-1]):
            if plot_type in ['prj', 'projection']:
                p = yt.ProjectionPlot(ds, normal=normal, fields=fields,  weight_field=weight_field, center=center, width = width, buff_size=buff_size )
            elif plot_type in ['slc', 'slice']:
                p = yt.SlicePlot(ds, normal=normal, fields=fields, center=center, width = width, buff_size=buff_size )                
            img = p.to_fits_data()[0].data.value
            if np.array_equal(normal, [0,1,0]):
                # rascas: -z, -x; yt z,x 
                img = np.flip(img, axis=(0,1) )
            elif np.array_equal(normal, [0,-1,0]):
                # rascas z, -x; yt z,x 
                img = np.flip(img, axis=1 )
            elif np.array_equal(normal, [0,0,1]):
                # rascas y, -x; yt x,y 
                img = np.transpose(img)
                img = np.flip(img, axis=1)
            elif np.array_equal(normal, [0,0,-1]):
                # rascas -y -x; yt x,y 
                img = np.transpose(img)
                img = np.flip(img, axis=(0,1) )
        # general cases, both rascas and yt use general orientation
        else:
            _, north_vector = num_util.cal_LOS_frame(normal, frame=another_code )
            if plot_type in ['prj', 'projection']:
                p = yt.ProjectionPlot(ds, normal=normal, north_vector=north_vector, fields=fields,  weight_field=weight_field, center=center, width = width, buff_size=buff_size )  
            elif plot_type in ['slc', 'slice']:
                p = yt.SlicePlot(ds, normal=normal, north_vector=north_vector, fields=fields, center=center, width = width, buff_size=buff_size ) 
            img = p.to_fits_data()[0].data.value
    return img

    

