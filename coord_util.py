from cProfile import label
from lib2to3.pygram import python_grammar_no_print_statement
import os
import re
import sys
import numpy as np 
import random
import matplotlib.pyplot as plt
import scipy 
import healpy

#####################
# data manipulation #
#####################
def rebin(a, *args, stat='mean'):
    '''rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    '''
    shape = a.shape
    lenShape = len(shape)
    factor = np.array(np.asarray(shape)/np.asarray(args), dtype=int)
    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],' % (i, i) for i in range(lenShape)] + \
             [')'] + ['.%s(%d)' %(stat, (i+1) ) for i in range(lenShape)]
    return eval(''.join(evList))

def get_unique(arr, log=False):
    arr = np.round(arr, 3)
    n_uniq= len( np.unique(arr) )
    print(n_uniq)
    return n_uniq 

def bin_from_array(arr):
    '''
    For a given array 'arr', generate bin array whose centre is the same as 'arr' (used for histogram).
    Note: 
        - arr should have equal intervals
    '''
    intervals = arr[1:] - arr[:-1]
    assert get_unique(intervals)==1
    
    interval = arr[1]-arr[0]
    
    nedges = len(arr)+1
    bins = np.zeros( nedges )
    bins[1:nedges-1] = 1/2*( arr[:-1] + arr[1:] )
    bins[0] = arr[0]-interval/2
    bins[-1] = arr[-1] + interval/2
    return bins 
    
def normalize_arr(arr, vmin=None, vmax=None, log=False):
    '''
    Normalize the array. Change from [vmin, vmax] to [0,1]
    '''
    if vmin is None:
        vmin = np.min(arr)
    if vmax is None:
        vmax = np.max(arr)
        
    if log==True:
        arr = np.log10(arr)
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)
    
    n_arr = (arr-vmin)/(vmax-vmin)
    n_arr[n_arr<0]=0
    n_arr[n_arr>1]=1
    
    return n_arr
    
def rev_normalize_arr(n_arr, vmin, vmax, log):
    '''
    Reverse the normalization of array. Change from [0,1] to [vmin, vmax]
    '''      
    if log==True:
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)
    
    arr = ( vmax-vmin )*n_arr + vmin
    
    if log==True:
        arr = 10**arr
        
    return arr

def make_vector_unit(v):
    '''
    Make a vector v to v_hat with unit length
    '''
    if type(v) is list:
        v = np.array(v)
    v_hat = v/np.sqrt( v.dot(v) ) # normalize
    return v_hat
        
####################
# array generation # 
####################
def non_uniform_array(bd_a, nbin_a, norm_a):
    '''
    generate non_uniform bins
    Input:
    bd_a: boundary array
    res_a: resolution array 
    '''
    bd_a  = np.array(bd_a)
    nbin_a = np.array(nbin_a)
    
    nseg = len(nbin_a)
    
    arr = np.array([])
    dL_a = bd_a[1:] - bd_a[:-1]
    
    for i, nbin, norm in zip( range(nseg), nbin_a, norm_a):
        if norm=='linear':
            sub_arr = np.linspace(bd_a[i], bd_a[i+1], nbin )
        elif norm=='log':
            sub_arr = np.logspace( np.log10(bd_a[i]), np.log10(bd_a[i+1]), nbin )
        else:
            raise Exception('not implemented.')
        
        if i<len(nbin_a)-1:
            arr = np.concatenate( (arr, sub_arr[:-1]), axis=0 )
        elif i==len(nbin_a)-1:
            arr = np.concatenate( (arr, sub_arr), axis=0 )
    return arr


######################
# LOS related function
######################

def generate_LOS_a(ndir=100, method='healpix'):
    if method=='healpix':
        NSIDE = int( np.sqrt(ndir/12) )
        NPIX = healpy.nside2npix(NSIDE)
        theta_a, phi_a= healpy.pix2ang(NSIDE, ipix=np.arange(NPIX) ) 
        LOS_a = np.zeros( (NPIX, 3 ) )
        LOS_a[:,0] = np.sin(theta_a)*np.cos(phi_a) 
        LOS_a[:,1] = np.sin(theta_a)*np.sin(phi_a) 
        LOS_a[:,2] = np.cos(theta_a)
        
    elif method=='x':
        if ndir==1:
            LOS_a = np.array( [ [1, 0, 0] ] )
        elif ndir==2:
            LOS_a = np.array( [ [1, 0, 0], [-1, 0, 0] ] )
        else:
            raise Exception('ndir not included')

    elif method=='xy':
        if ndir==2:
            LOS_a = np.array( [ [1., 0., 0.], [0., 1., 0.] ] ) 
        elif ndir==4:
            LOS_a = np.array( [ [1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.] ] ) 
        else:
            raise Exception('ndir not included')

    elif method=='xyz':
        if ndir==3:
            LOS_a = np.array( [ [1., 0., 0.], [0., 1., 0.], [0., 0., 1.] ] ) 
        elif ndir==6:
            LOS_a = np.array( [ [1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.] ] ) 
        else:
            raise Exception('ndir not included')

    elif method=='random':
        # random direction
        LOS_a = []
        for i in range(ndir):
            theta = random.random()*np.pi
            phi = random.random()*2.*np.pi
            LOS_a.append( [ np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta) ] )
        LOS_a = np.array(LOS_a)

    elif method=='plane':
        # uniformly sample direction through a point in a plane, currently only plane perp to y axis
        LOS_a = []
        theta_a = np.linspace(0, 2*np.pi, ndir+1)[:-1]
        for i, theta in enumerate(theta_a):
            LOS_a.append( [np.cos(theta), 0, np.sin(theta)] )
        LOS_a = np.array(LOS_a) 
        
    elif method=='manual' and ndir==1:
        # for Callum Nature paper
        LOS_a = [ [-0.7845, 0.0, 0.620] ]
        
    return LOS_a

def where_row_in_array(row , arr, thres=1e-4):
    # return the index of the 1d row in a 2d arr
    dev_a = np.sum( np.abs(arr-row), axis=1) / np.sum( np.abs(arr), axis=1)
    idx_a = np.where( dev_a < thres )[0]
    if len(idx_a)==0:
        print('func where_row_in_array: no such row')
        return None
    elif len(idx_a)==1:
        print('func where_row_in_array: only one row in the array')
        idx = idx_a[0]
        return idx
    else:
        print('func where_row_in_array: multiple rows')
        return idx_a

def find_xy_in_healpix(ndir):
    '''
    return the index of xyz axis from healpix LOS array
    '''
    LOS_a = generate_LOS_a(ndir=ndir, method='healpix')
    ix = where_row_in_array( [1,0,0], LOS_a, thres=1e-10)
    iy = where_row_in_array( [0,1,0], LOS_a, thres=1e-10)
    return ix, iy

def LOS2angle(LOS_a):
    '''
    convert LOS to its theta and phi.
    Input:
        LOS_a
    Output:
        theta_a, from -pi/2 to pi/2
        phi_a, from -pi to pi
    test: [1,0,0] -> [pi/2, 0], [0,1,0] -> [pi/2, pi/2]
    '''
    theta_a = []
    phi_a = []
    for LOS in LOS_a:
        sintheta = np.sqrt(LOS[0]**2 + LOS[1]**2 )
        theta = np.arctan2( sintheta, LOS[2] )
        phi = np.arctan2( LOS[1], LOS[0] )
        theta_a.append(theta)
        phi_a.append(phi)
    return theta_a, phi_a

def cal_LOS_frame(LOS, frame='default'):
    '''
    Convention in RASCAS:
    unit vector for the axis, defined in the same way as kobs_perp_1,2 in module_mock.f90, 
    Input:
        LOS
    '''
    LOS = make_vector_unit(LOS)
    if frame=='default':
        if np.abs(LOS[0])<1:
            # k_perp1 = k_obs cross xhat then normalize to unit; k_perp2 = kobs cross k_perp1 
            # special case:
            #   LOS=yhat, k_perp1=-zhat, k_perp2=-xhat
            #   LOS=zhat, k_perp1=yhat,  k_perp2=-xhat
            k_perp1 = np.cross(LOS, [1,0,0] )
            k_perp1 = make_vector_unit(k_perp1) 
            k_perp2 = np.cross(LOS, k_perp1)
        else:
            # if k_obs = \pm xhat, k_perp1 = yhat, kperp2 = zhat
            k_perp1=[0,1,0]
            k_perp2=[0,0,1]
    else:
        raise Exception('%s frame is not implemented'%frame )
    
    print('cal_LOS_frame: k_perp1,', k_perp1, 'k_perp2', k_perp2)
    return k_perp1, k_perp2
    
def cal_projection_pos(x_mesh, y_mesh, z_mesh, LOS):
    # projection map along arbitrary LOS

    k_perp1, k_perp2 = cal_LOS_frame(LOS)    
    
    # assign projected coordinate wx wy for each cell
    wx = x_mesh*k_perp1[0]+ y_mesh*k_perp1[1]+ z_mesh*k_perp1[2]
    wy = x_mesh*k_perp2[0]+ y_mesh*k_perp2[1]+ z_mesh*k_perp2[2]
    return wx, wy

def cal_proj_vel(vx_mesh, vy_mesh, vz_mesh, LOS):
    v_los = vx_mesh*LOS[0] + vy_mesh*LOS[1] + vz_mesh*LOS[2]
    v_los = -v_los
    return v_los

def rotate_yt_image(image, axis):
    '''
    rotate yt image to make it have the same frame as described in cal_LOS_frame default case.
    '''
    if axis=='x':
        image = np.rot90(image, k=1, axes=(1,0) )
    elif axis=='y':
        image = np.rot90(image, k=1, axes=(0,1))
    elif axis=='z':
        raise Exception('not implemented')
    return image 


'''
Coordinate system
'''
def create_cartesian_coord(info, size=10, ndim=3, center=True):
    '''
    Create cartesian coordinates
    '''
    if ndim==1:
        nx = info['nx']
        if center:
            nx = nx+1
        x = np.linspace(-size/2, size/2, nx)
        if center:
            x = 1/2*(x[:-1] + x[1:] )
        return x

    elif ndim==2:
        nx = info['nx']
        ny = info['ny']
        if center==True:
            nx = nx+1
            ny = ny+1        
        x = np.linspace(-size/2,size/2, nx)
        y = np.linspace(-size/2,size/2, ny)
        if center==True:
            x = 1/2*(x[:-1] + x[1:] )
            y = 1/2*(y[:-1] + y[1:] )
        img_x, img_y = np.meshgrid( x,y, indexing='ij')
        return img_x, img_y
    
    if ndim==3:
        nx = info['nx']
        ny = info['ny']
        nz = info['nz']
        if center==True:
            nx +=1
            ny +=1
            nz +=1
        x = np.linspace(-size/2,size/2, nx)
        y = np.linspace(-size/2,size/2, ny)
        z = np.linspace(-size/2,size/2, nz)
        if center==True:
            x = 1/2*(x[:-1] + x[1:] )
            y = 1/2*(y[:-1] + y[1:] )
            z = 1/2*(z[:-1] + z[1:] )        

        cube_x, cube_y, cube_z = np.meshgrid( x,y,z, indexing='ij')
        return cube_x, cube_y, cube_z

create_coord = create_cartesian_coord

def polar_coord(x_mesh, y_mesh, center=[0,0]):
    # given the center, calculate two angles in polar coordinate of cells. 
    x_mesh -= center[0]
    y_mesh -= center[1]
    rho_mesh = np.sqrt(x_mesh**2 + y_mesh**2)
    phi_mesh = np.arctan2(y_mesh, x_mesh)
    return rho_mesh, phi_mesh


# spherical coordinate
def unit_vec_sph_coord( rvec, rvec_coord = 'xyz' ):
    '''
    Unit vector of spherical coordinate, in cartesian coordinate.
    Inputs:
        rvec
        rvec_coord: the coordinate system rvec is written in
    '''
    if rvec_coord == 'xyz':
        x,y,z = rvec
        r = np.sqrt(x**2+y**2+z**2)
        rho = np.sqrt(x**2+y**2)
        r_hat = rvec/r 
        theta_hat = np.array([x*z, y*z, -rho**2 ]) / (rho*r)
        phi_hat = np.array([-y, x, 0]) / np.sqrt(x**2+y**2)

    elif rvec_coord == 'rtp':
        r, theta, phi = rvec 
        r_hat = np.array([ np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta) ])
        theta_hat = np.array([ np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(phi) ])
        phi_hat = np.array( np.sin(phi), np.cos(phi), 0 )

    return r_hat, theta_hat, phi_hat

def cal_pos_sph_coord(x_mesh, y_mesh, z_mesh, center=[0,0,0]):
    # given the center, calculate two angles in spherical coordinate of cells. 
    x_mesh -= center[0]
    y_mesh -= center[1]
    z_mesh -= center[2]
    r_mesh = np.sqrt( x_mesh**2 + y_mesh**2 + z_mesh**2  )
    theta_mesh = np.arctan( np.sqrt( x_mesh**2+y_mesh**2) / z_mesh )
    phi_mesh = np.arctan2( y_mesh, x_mesh )

    # theta phi will be nan at (0,0,0)
    theta_mesh[theta_mesh!=theta_mesh] = 0 
    phi_mesh[phi_mesh!=phi_mesh] = 0 

    # change theta -pi/2 -> pi/2, phi -pi -> pi to
    # theta 0 -> pi, phi 0 -> 2pi
    theta_mesh += np.pi/2
    phi_mesh[phi_mesh<0] += 2*np.pi
    return r_mesh, theta_mesh, phi_mesh

def cal_vector_in_sph_rtp_frame( v, rvec, rvec_coord):
    '''
    Given a vector with origin at pos in cartesian coordinate, calcalate its component in spherical rhat, theta_hat, phi_hat frame.
    '''
    r_hat, theta_hat, phi_hat = unit_vec_sph_coord( rvec, rvec_coord )
    v_r = np.dot(v, r_hat) 
    v_theta = np.dot(v, theta_hat)
    v_phi = np.dot(v, phi_hat)
    return v_r, v_theta, v_phi

def cal_vector_in_r(x, y, z, vx, vy, vz):
    r = np.sqrt( x**2 + y**2 + z**2 )
    vr = (vx*x + vy*y + vz*z) / r
    return vr


# cylindrical coordiate
def cal_vector_in_cyl_coord():
    pass 