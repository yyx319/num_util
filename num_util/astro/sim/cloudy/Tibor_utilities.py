#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:17:06 2023

@author: tibor
"""

import subprocess
import multiprocessing as mp
import numpy as np
from scipy.special import erfc
import config
import warnings
from config import updateZColGrids
from scipy.integrate import quad
from scipy.spatial.transform import Rotation as R
from print_msg import print_status
from painting import makeGridWithSPHNoPBC
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from plotting_utilities import vizZAndHI
import yt
import os
rank = 0


def fixSFR(M_star, SFR, z_vals, zrange):
    # Restrict the data to the given redshift range
    zwhere = np.logical_and(z_vals < zrange[1], z_vals > zrange[0])
    M_star = M_star[zwhere]
    SFR = SFR[zwhere]
    z_vals = z_vals[zwhere]

    # Identify the zero SFR indices
    zero_indices = np.where(SFR == 0)[0]

    for idx in zero_indices:
        start = idx
        end = idx
        while end + 1 < len(SFR) and SFR[end + 1] == 0:
            end += 1

        left_value = SFR[start - 1] if start > 0 else None
        right_value = SFR[end + 1] if end + 1 < len(SFR) else None

        if left_value is not None and right_value is not None:
            # Linear interpolation between the non-zero neighbors
            interp_values = np.linspace(left_value, right_value, end - start + 2)[1:-1]
        elif left_value is not None:
            # Extend the left value if there's no right neighbor
            interp_values = np.full(end - start + 1, left_value)
        elif right_value is not None:
            # Extend the right value if there's no left neighbor
            interp_values = np.full(end - start + 1, right_value)
        else:
            # If both neighbors are None, leave as zero
            interp_values = np.zeros(end - start + 1)

        SFR[start:end + 1] = interp_values

    return M_star, SFR, z_vals

def fixSFROld(M_star, SFR, z_vals, zrange):
    zwhere = np.logical_and(z_vals < zrange[1], z_vals > zrange[0])
    M_star = M_star[zwhere]
    SFR = SFR[zwhere]
    z_vals = z_vals[zwhere]
    # Set SFR = 0 to average of neighboring values, divided by two
    zero_indices = np.where(SFR == 0)[0]
    for idx in zero_indices:
        # Find the start and end of the zero segment
        if SFR[idx] == 0:
            start = idx
            end = idx
            while end + 1 < len(SFR) and SFR[end + 1] == 0:
                end += 1
            # Calculate the average of the non-zero neighbors
            left_value = SFR[start - 1] if start > 0 else None
            right_value = SFR[end + 1] if end + 1 < len(SFR) else None
            if left_value is not None and right_value is not None:
                average_value = (left_value + right_value) / 4.0
            elif left_value is not None:
                average_value = left_value / 2.0
            elif right_value is not None:
                average_value = right_value / 2.0
            else:
                average_value = 0
            SFR[start:end + 1] = average_value
    return M_star, SFR, z_vals

def getSFH(start_time, scale_factor_myrs, tmid, gx_star_masses, gx_initial_star_masses, cosmic_time_stars, delta_t_target):
    # star_masses, star_initial_masses in internal ART units = g
    # scale_factor_myrs, cosmic_time_stars and tmid in Myr
    # return in Msun / yr and Msun
    
    # Constants
    solar_mass = 1.989 * 10**(33) # in grams
    
    # Initialize
    tmin = tmid - delta_t_target/2
    tmax = tmid + delta_t_target/2
    which_idxs = np.where(np.logical_and(cosmic_time_stars>tmin, cosmic_time_stars<tmax) == True)[0]
    which_idxs_older_equal = np.where((cosmic_time_stars<tmax) == True)[0]
    slope = (gx_star_masses[which_idxs_older_equal]/solar_mass-gx_initial_star_masses[which_idxs_older_equal]/solar_mass)/(scale_factor_myrs-cosmic_time_stars[which_idxs_older_equal])
    print_status(rank, start_time, "Note: t_min is {:.2e} Myr and t_max is {:.2e} Myr. Have {} stellar particles. Slope is {}".format(tmin, tmax, len(gx_star_masses), slope))
    gx_initial_star_masses_in_bin = gx_initial_star_masses[which_idxs]/solar_mass
    sfh = np.sum(gx_initial_star_masses_in_bin)/((tmax-tmin)*10**6)  # Though here initial vs surviving stellar mass differences are tiny, since timebins are right behind snapshot
    stellar_mass = np.sum(gx_initial_star_masses[which_idxs_older_equal]/solar_mass+slope*(tmid-cosmic_time_stars[which_idxs_older_equal]))
    return sfh, stellar_mass

def magnetothermal_jeans_length(c_sound, pth, rho_gas, BlxID, BrxID, BlyID, BryID, BlzID, BrzID, dfac, tfac):
    # c_sound in cm/s
    # pth in dyn/cm**2
    # rho_gas in g/cm^3
    # all B fields in Gauss = cm−1/2⋅g1/2⋅s−1
    Ggrav = 6.67430e-8  # Gravitational constant in cgs units, i.e. dyn⋅cm2⋅g−2, dyn is g⋅cm/s2
    c_sound_eff = effective_magnetic_sound_speed(c_sound, pth, BlxID, BrxID, BlyID, BryID, BlzID, BrzID)
    lambda_magthermal = c_sound_eff * np.sqrt(np.pi) / np.sqrt(Ggrav * rho_gas) # no turbulence assumed here
    return lambda_magthermal # should be in cm

def effective_magnetic_sound_speed(c_sound, pth, BlxID, BrxID, BlyID, BryID, BlzID, BrzID):
    # c_sound in cm/s
    # pth in dyn/cm**2
    # all B fields in Gauss = cm−1/2⋅g1/2⋅s−1
    p_mag = 0.5 * total_magfield_sq(BlxID, BrxID, BlyID, BryID, BlzID, BrzID) # should be in dyn/cm**2
    inverse_beta_plasma = p_mag / pth
    effective_magnetic_c_sound = c_sound * np.sqrt(1.0 + inverse_beta_plasma)
    return effective_magnetic_c_sound # should be in cm/s

def total_magfield_sq(BlxID, BrxID, BlyID, BryID, BlzID, BrzID):
    # all B fields in Gauss = cm−1/2⋅g1/2⋅s−1
    Btot = 0.25 * ((BlxID + BrxID)**2 +
                     (BlyID + BryID)**2 +
                       (BlzID + BrzID)**2)
    return Btot # should be in dyn/cm**2 = Gauss**2

def calculate_local_sfe(t0, dx, pth, rho_gas, BlxID, BrxID, BlyID, BryID, BlzID, BrzID, c_sound, tfac, dfac):
    # dx in cm
    # pth in dyn/cm**2
    # rho_gas in g/cm^3
    # c_sound in cm/s
    # all B fields in Gauss = cm−1/2⋅g1/2⋅s−1
    # tfac and dfac are time and distance code unit to physical unit conversion factors, respectively
    print_status(rank, t0, "Calling calculate_local_sfe()")
    Ggrav = 6.67430e-8  # Gravitational constant in cgs units, i.e. dyn⋅cm2⋅g−2, dyn is g⋅cm/s2
    local_sfe = np.zeros_like(dx)
    lambda_dummy = magnetothermal_jeans_length(c_sound, pth, rho_gas, BlxID, BrxID, BlyID, BryID, BlzID, BrzID, dfac, tfac)
    pmag_dummy = 0.5 * total_magfield_sq(BlxID, BrxID, BlyID, BryID, BlzID, BrzID) # dyn/cm**2
    invbeta_dummy = pmag_dummy / pth
    csound_dummy = c_sound**2 * (1.0 + invbeta_dummy) # cm/s
    trgv = 0.0  # Approximating trgv by cell velocity
    alpha0 = 5.0 / (np.pi * Ggrav * rho_gas) * (trgv + csound_dummy) / dx**2 # unitless
    e_cts, phi_t, theta = 0.5, 0.57, 0.33
    betafunc = ((1.0 + 0.925 * invbeta_dummy**1.5)**(2/3)) / ((1.0 + invbeta_dummy)**2) # unitless
    scrit = np.log(0.067 * betafunc / (theta**2) * alpha0 * trgv / csound_dummy) # unitless
    sigs = np.log(1.0 + 0.16 * trgv / csound_dummy) # σs^2 unitless, dispersion of the logarithm of the gas density to the mean gas density
    ydummy = e_cts / 2.0 * phi_t * np.exp(3.0 / 8.0 * sigs) * (2.0 - erfc((sigs - scrit) / np.sqrt(2.0 * sigs)))
    
    # There is no star formation in cells where \Delta x cell < λ_J, MTT. Also make sure only minimal-size cells are considered (otherwise they are not star-forming)
    local_sfe[np.logical_and(dx > lambda_dummy, dx < 1.5 * dx.min())] = ydummy[np.logical_and(dx > lambda_dummy, dx < 1.5 * dx.min())]

    return local_sfe # unitless

def project(start_time, gas_xyz, gas_masses, V, sfe, params):
    """
    :param start_time: for printing purposes
    :type start_time: float
    :param gas_xyz: gas cell positions in units of pkpc/h
    :type gas_xyz: (N,3) floats
    :param gas_masses: gas cell masses in code units (10^10 M_sun/h)
    :type gas_masses: (N,) floats
    :param V: cell volume in units of (pkpc/h)**3
    :type V: (N,) floats
    :param sfe: star formation efficiency of each cell, unitless, or whatever else you want to project
    :type sfe: (N,) floats
    :param params: various numerical and physical params for the model
    :type params: tuple
    """
    L_BOX, theta, phi, RAD, HUBBLE, N_grid = params
    # COM
    gas_xyz = respectPBCNoRef(gas_xyz, L_BOX) # in pkpc/h
    com = calcMode(gas_xyz, gas_masses, rad=1000) # rad in pkpc/h
    print_status(rank, start_time, "Mode of gx is {} pkpc/h".format(com))
        
    # Define normal vector orthographic projection grid
    norm_vec = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    a = norm_vec[0]
    b = norm_vec[1]
    c = norm_vec[2]
    denom = np.sqrt(a**2+b**2+c**2) # Should be 1 anyway
    cost = c/denom
    sint = np.sqrt(a**2+b**2)/denom
    u1 = b/np.sqrt(a**2+b**2)
    u2 = -a/np.sqrt(a**2+b**2)
    rot_matrix = np.array([[cost+u1**2*(1-cost), u1*u2*(1-cost), u2*sint],
                       [u1*u2*(1-cost), cost+u2**2*(1-cost), -u1*sint],
                       [-u2*sint, u1*sint, cost]])
    r1 = R.from_matrix(rot_matrix)
    
    proj_vecs = np.zeros((len(gas_xyz), 3), dtype = np.float32)
    for i, xyz in enumerate(gas_xyz):
        proj_vecs[i] = (xyz - com)/(HUBBLE/100) - norm_vec*(np.dot((xyz-com)/(HUBBLE/100), norm_vec)) # physical kpc
    # Translate such that plane passes through origin, which is the case already, but just in case
    d = -(a*proj_vecs[0,0]+b*proj_vecs[0,1]+c*proj_vecs[0,2]) # One could take any point in plane, not only the one with index 0
    proj_vecs[:, 0] = proj_vecs[:, 0] - a * d
    proj_vecs[:, 1] = proj_vecs[:, 1] - b * d
    proj_vecs[:, 2] = proj_vecs[:, 2] - c * d
    # Rotate vectors in plane onto plane parallel to xy plane (z=const=0 plane)
    proj_xy = r1.apply(proj_vecs) # z-coordinates should be const=0
    # Translation in plane
    proj_xy = proj_xy + RAD # center should become left-bottom edge of grid, physical kpc
    # Discard gas cells outside of enlarged, square-shaped "aperture"
    keep_x = np.logical_and(proj_xy[:,0] >= 0.0, proj_xy[:,0] < 2*RAD)
    keep_y = np.logical_and(proj_xy[:,1] >= 0.0, proj_xy[:,1] < 2*RAD)
    keep = np.logical_and(keep_x, keep_y)
    proj_xy = proj_xy[keep] # already pkpc, so it matches hsml units
    gas_to_add = np.arange(len(gas_xyz))
    gas_to_add = gas_to_add[keep]
    
    # Prepare cubic spline kernel deposition onto grid
    r_cell = (3*V[gas_to_add]/(4*np.pi)/(HUBBLE/100)**3)**(1/3) # physical kpc
    hsml = np.float32(2.5*r_cell) # physical kpc
    
    mass_weighted_sfe = np.float32(sfe*gas_masses)
    sfemass_grid = makeGridWithSPHNoPBC(np.float32(proj_xy[:,0]), np.float32(proj_xy[:,1]), np.float32(mass_weighted_sfe[gas_to_add]), np.float32(hsml), np.float32(RAD*2), N_grid) # 10^10 M_sun/h / pkpc**2
    gasmass_grid = makeGridWithSPHNoPBC(np.float32(proj_xy[:,0]), np.float32(proj_xy[:,1]), np.float32(gas_masses[gas_to_add]), np.float32(hsml), np.float32(RAD*2), N_grid) # 10^10 M_sun/h / pkpc**2
    gasmass_grid[gasmass_grid < 1e-15] = 1e-15 # to avoid nans after division by zero
    sfe_grid = sfemass_grid/gasmass_grid # dimensionless sfe grid
    return sfe_grid

def getBirthTime(sim, star_age, HUBBLE, OMEGA_R, OMEGA_M, OMEGA_L):
    Mpc_in_km = 3.0856775812*10**19
    Megayrs_in_s = 31536000000000 # i.e. 31536 Giga seconds
    H0_Myr = HUBBLE/Mpc_in_km*Megayrs_in_s # 1/Myr
    tH = 1/H0_Myr # Myr
    if "RT" not in sim:
        a_s = np.logspace(-3,0,1000)
        sconformals = np.zeros_like(a_s)
        for ind, a in enumerate(a_s):
            sconformals[ind] = getSuperComovingTime(a, HUBBLE/100, OMEGA_R, OMEGA_M, OMEGA_L)*H0_Myr # unitless, measured from z = 0, thus negative values
        conformals_interp = interp1d(sconformals, a_s, bounds_error = False, fill_value = 'extrapolate')
        # Convert supercomoving time (tau) to cosmic time (t)
        cosmic_times = np.zeros_like(star_age)
        for i, tau in enumerate(star_age):
            a = conformals_interp(tau) # tau is negative
            cosmic_times[i] = getCosmicTime(a, HUBBLE / 100, OMEGA_R, OMEGA_M, OMEGA_L)
        births = cosmic_times # lookforward time in Myr
    else:
        births = np.array(star_age)*tH + getCosmicTime(1, HUBBLE / 100, OMEGA_R, OMEGA_M, OMEGA_L) # lookforward time in Myr
    return births

def getSuperComovingTime(a, h, Omega_R, Omega_M, Omega_L): # Need to avoid singularity at a = 0 (BB)
    def integrand(a_prime, Omega_R, Omega_M, Omega_L):
        return 1/(a_prime**3*np.sqrt(Omega_R*a_prime**(-4) + Omega_M*a_prime**(-3) + Omega_L))
    H0 = 100*h # km/s/Mpc
    Mpc_in_km = 3.0856775812*10**19
    Megayrs_in_s = 31536000000000 # i.e. 31536 Giga seconds
    H0 = H0/Mpc_in_km*Megayrs_in_s # 1/Myrs
    return quad(integrand, 1, a, args=(Omega_R, Omega_M, Omega_L))[0]/H0 # in Myrs, tau = 0 at z = 0

def readTracker(track_file, i_track_choose, pc, remove_fmergers):
    # Define lists to store the coordinates of galaxy tracks
    if i_track_choose == 0:
        warnings.warn("You are calling readTracker() with i_track_choose equal to zero. Better values would be 1, 2 or 3. Are you sure?")
    x_tracks = []
    y_tracks = []
    z_tracks = []
    rvirs = []
    i_tracks = []
    merging = []
    actives = [] # Only needed for old tracker files, but this is not returned by this function
    # Find out whether tracker file is old or new
    with open(track_file, "r") as file:
        # Read all lines
        lines = file.readlines()
        if lines[0].strip() == "Header v4.4":
            new_tracker = True
        else:
            new_tracker = False
    # Open the file
    if new_tracker:
        with open(track_file, "r") as file:
            # Read all lines
            lines = file.readlines()
            # Iterate through the lines
            i = 0
            while i < len(lines):
                if lines[i].startswith("unit_l"):
                    # Split the line into components and get the part after the "=" sign
                    unit_l = float(lines[i].split("=")[1].strip())
                elif lines[i].startswith("unit_d"):
                    unit_d = float(lines[i].split("=")[1].strip())
                elif lines[i].startswith("unit_t"):
                    unit_t = float(lines[i].split("=")[1].strip())
                if lines[i].startswith("itrack"):
                    parts = lines[i].split("=")[1].strip().split()
                    i_track_values = [np.int32(part) for part in parts][1]
                    i_tracks.append(i_track_values)
                if lines[i].startswith("merging_to"):
                    parts = lines[i].split("=")[1].strip().split()
                    merging_to = [np.int32(part) for part in parts][0]
                    merging.append(merging_to)
                if lines[i].startswith("merged_to"):
                    i_tracks = i_tracks[:-1] # Remove tracker from list
                # Check if the line contains xtrack information
                if lines[i].startswith("xpos_track"):
                    parts = lines[i].split("=")[1].strip().split()
                    x_tracks.append(float(parts[0]))
                    y_tracks.append(float(parts[1]))
                    z_tracks.append(float(parts[2]))
                if lines[i].startswith("rvir_halo"): # careful, this is pkpc
                    parts = lines[i].split("=")[1].strip().split()
                    rvirs.append(float(parts[1]))
                    if remove_fmergers and len(merging) > 0 and merging[-1] != -1:
                        # Remove entry
                        i_tracks = i_tracks[:-1]
                        x_tracks = x_tracks[:-1]
                        y_tracks = y_tracks[:-1]
                        z_tracks = z_tracks[:-1]
                        rvirs = rvirs[:-1]
                        merging = merging[:-1]
                if lines[i].startswith("Tracked particle IDs"):
                    break
                # Move to the next line
                i += 1
        rvirs = np.array(rvirs)*(pc*1000)/unit_l # in code length now, assuming rvirs was in pkpc before this line (not ckpc)
    if len(rvirs) == 0 or (not new_tracker): # Old tracker files
        rvirs = []; i_tracks = []
        with open(track_file, "r") as file:
            # Read all lines
            lines = file.readlines()
            # Iterate through the lines
            i = 0
            while i < len(lines):
                if lines[i].startswith("unit_l"):
                    unit_l = float(lines[i].split("=")[1].strip())
                elif lines[i].startswith("unit_d"):
                    unit_d = float(lines[i].split("=")[1].strip())
                elif lines[i].startswith("unit_t"):
                    unit_t = float(lines[i].split("=")[1].strip())
                if lines[i].startswith("itrack"):
                    i_track = np.int32(lines[i].split("=")[1].strip())
                    i_tracks.append(i_track)
                if lines[i].startswith("active"):
                    active = np.int32(lines[i].split("=")[1].strip())
                    actives.append(active)
                # Check if the line contains xtrack information
                if lines[i].startswith("xtrack"):
                    x_track = float(lines[i].split("=")[1].strip())
                    y_track = float(lines[i + 1].split("=")[1].strip())
                    z_track = float(lines[i + 2].split("=")[1].strip())
                    x_tracks.append(x_track)
                    y_tracks.append(y_track)
                    z_tracks.append(z_track)
                if lines[i].startswith("rvir"):
                    rvir = float(lines[i].split("=")[1].strip())
                    rvirs.append(rvir)
                    """if len(actives) > 0 and actives[-1] != 1:
                        # Remove entry
                        i_tracks = i_tracks[:-1]
                        x_tracks = x_tracks[:-1]
                        y_tracks = y_tracks[:-1]
                        z_tracks = z_tracks[:-1]
                        rvirs = rvirs[:-1]
                        actives = actives[:-1]"""
                    i += 10 # Move to the next set of coordinates for the next galaxy
                if lines[i].startswith("Tracked particle IDs"):
                    break
                # Move to the next line
                i += 1
    x_tracks = np.array(x_tracks); y_tracks = np.array(y_tracks); z_tracks = np.array(z_tracks)
    rvirs = np.array(rvirs); i_tracks = np.array(i_tracks)
    return x_tracks, y_tracks, z_tracks, rvirs, i_tracks, unit_l, unit_d, unit_t

def define_loading_fields_funct(select):
    # This function defines the proper load fields to upload to memory
    # Use this in combination with the yt.load function
    # ds = yt.load(readf_route, fields=load_fields, extra_particle_fields=extra_particle_fields)
    load_fields=[]
    if(select in ['HD+SfFb','HD+SfFbBoost','HD+SfNoFb','HD+Sf',
                  'HD+thSfFb','HD+thSfFbBoost','HD+thSfNoFb','HD+thSf']):
        load_fields=["Density",
                     "x-velocity","y-velocity","z-velocity",
                     "Pressure","refine","Metallicity"]
    elif (select in ['RT+SfFb', 'RT', 'RT+Boost'] ):
        load_fields=["Density",
                     "x-velocity","y-velocity","z-velocity",
                     "Pressure","refine","Metallicity",
                     "xHII","xHeII","xHeIII"
                     ]
    elif (select in ['MHD+SfFb','iMHD+SfFb','sMHD+SfFb', 'MHD','iMHD','sMHD']):
        load_fields=["Density",
                     "x-velocity","y-velocity","z-velocity",
                     "x-Bfield-left","y-Bfield-left","z-Bfield-left",
                     "x-Bfield-right","y-Bfield-right","z-Bfield-right",
                     "Pressure","refine","Metallicity"]
    elif (select in ['CRiMHD+SfFb', 'CRiMHD'] ):
        load_fields=["Density",
                     "x-velocity","y-velocity","z-velocity",
                     "x-Bfield-left","y-Bfield-left","z-Bfield-left",
                     "x-Bfield-right","y-Bfield-right","z-Bfield-right",
                     "CR_Pressure","Pressure","refine","Metallicity"]
    elif (select in ['RTiMHD+SfFb','RTsMHD+SfFb', 'RTiMHD','RTsMHD', 'RTiMHD-new']):
        load_fields=["Density",
                     "x-velocity","y-velocity","z-velocity",
                     "x-Bfield-left","y-Bfield-left","z-Bfield-left",
                     "x-Bfield-right","y-Bfield-right","z-Bfield-right",
                     "Pressure","refine","Metallicity","xHII","xHeII","xHeIII"]
    elif (select in ['RTCRiMHD+SfFb','RTnsCRiMHD+SfFb', 'RTCRiMHD','RTnsCRiMHD']):
        load_fields=["Density",
                     "x-velocity","y-velocity","z-velocity",
                     "x-Bfield-left","y-Bfield-left","z-Bfield-left",
                     "x-Bfield-right","y-Bfield-right","z-Bfield-right",
                     "CR_Pressure","Pressure","refine","Metallicity",
                     "xHII","xHeII","xHeIII"]
    else:
        print( "Not found load type, loading default")
        load_fields=["Density",
                     "x-velocity","y-velocity","z-velocity",
                     "Pressure"]
    return load_fields

def extractZs(sim):
    if sim == "RTiMHD":
        snaps = [x for x in np.int32(np.linspace(10, 67, 58))]
    elif sim == "iMHD":
        snaps = [x for x in np.int32(np.linspace(10, 67, 58))]
    else:
        assert sim == "RTnsCRiMHD"
        snaps = [x for x in np.int32(np.linspace(10, 65, 56))]
    if os.path.exists('{}/zs_{}.txt'.format(config.CAT_DIR, sim)):
        pass
    else:
        zs = np.zeros((len(snaps), ), dtype = np.float32)
        for snap_i, snap in enumerate(snaps):
            try:
                ds = yt.load("/data/ERCblackholes4/smartin/Spiral1/{}/output_{:05d}/info_{:05d}.txt".format(sim, snap, snap))
                z = ds.current_redshift
                zs[snap_i] = z
            except:
                pass
        np.savetxt('{}/zs_{}.txt'.format(config.CAT_DIR, sim), zs, fmt='%1.7e')
    return snaps

def findMode(xyz, masses, rad):
    """ Find mode of point distribution xyz
    
    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N^3x3) floats
    :param masses: masses of particles of type 1 or type 4
    :type masses: (N^3x1) floats
    :param rad: initial radius to consider away from COM of object
    :type rad: float
    :return: mode of point distribution
    :rtype: (3,) floats"""
    com = np.sum(xyz*np.reshape(masses, (masses.shape[0],1)), axis = 0)/masses.sum()
    distances_all = np.linalg.norm(xyz-com,axis=1)
    xyz_constrain = xyz[distances_all < rad]
    masses_constrain = masses[distances_all < rad]
    if xyz_constrain.shape[0] < 5: # If only < 5 particles left, return
        return com
    else:
        rad *= 0.83 # Reduce radius by 17 %
        return findMode(xyz_constrain, masses_constrain, rad)
    
def getCosmicTime(a, h, Omega_R, Omega_M, Omega_L):
    # Based on https://physics.stackexchange.com/questions/92805/what-is-the-equation-for-the-scale-factor-of-the-universe-at-for-the-best-fi
    def integrand(a_prime, Omega_R, Omega_M, Omega_L):
        return a_prime/np.sqrt(Omega_R + Omega_M*a_prime + Omega_L*a_prime**4)
    H0 = 100*h # km/s/Mpc
    Mpc_in_km = 3.0856775812*10**19
    Megayrs_in_s = 31536000000000 # i.e. 31536 Giga seconds
    H0 = H0/Mpc_in_km*Megayrs_in_s # 1/Myrs
    return 1/H0*quad(integrand, 0.0, a, args=(Omega_R, Omega_M, Omega_L))[0] # in Myrs

def getOmegaR(T_0, h):
    # Based on https://physics.stackexchange.com/questions/94181/where-is-radiation-density-in-the-planck-2013-results
    a_B = 7.56577*10**(-16) # Jm^{-3}K^{-4}, J = kg m^2 s^{-2}
    c = 299792458 # m/s
    rho_gamma = a_B*T_0**4/c**2
    rho_nu = 3.046*7/8*(4/11)**(4/3)*rho_gamma
    rho_c_0 = 1.87847*h**2*10**(-26) # kg m^{-3}
    Omega_R = rho_gamma/rho_c_0 + rho_nu/rho_c_0 # around 9.23640×10-5
    return Omega_R
    
# Calzetti 1994, albedo
def getAlbedo(lambda_):
    # lambda_ in Angstrom
    y = np.log10(lambda_)
    if hasattr(y, "__len__"):
        to_return = np.zeros((len(y),), dtype = np.float32)
        has_len = True
    else:
        to_return = np.zeros((1,), dtype = np.float32)
        y = np.array([y])
        lambda_ = np.array([lambda_])
        has_len = False
    for i in range(len(y)):
        if lambda_[i] >= 1000 and lambda_[i] <= 3460:
            to_return[i] = 0.43+0.366*(1-np.exp(-(y[i]-3)**2/0.2))
        elif lambda_[i] > 3460 and lambda_[i] <= 10000:
            to_return[i] = -0.48*y[i]+2.41
        else:
            to_return[i] = 1.0
    if has_len:
        return to_return
    else:
        return to_return[0]
    
# Calzetti 1994, scattering anisotropy weight factor
def getAnisoScatt(lambda_):
    # lambda_ in Angstrom
    y = np.log10(lambda_)
    if hasattr(y, "__len__"):
        to_return = np.zeros((len(y),), dtype = np.float32)
        has_len = True
    else:
        to_return = np.zeros((1,), dtype = np.float32)
        y = np.array([y])
        lambda_ = np.array([lambda_])
        has_len = False
    for i in range(len(y)):
        if lambda_[i] >= 1000 and lambda_[i] <= 10000:
            to_return[i] = 1 - 0.561*np.exp(-(abs(y[i]-3.3112)**(2.2))/0.17)
        else:
            to_return[i] = 0.0
    if has_len:
        return to_return
    else:
        return to_return[0]
    
# Cardelli 1989
def getSolarExtinction(lambda_):
    # lambda_ in Angstrom
    R_v = 3.1 # Value in diffuse interstellar medium (ISM), e.g. in some dense clouds you find R_v = 5
    x = 10000/lambda_ # in micrometer inverse
    if hasattr(lambda_, "__len__"):
        a = np.zeros((len(x),), dtype = np.float32)
        b = np.zeros((len(x),), dtype = np.float32)
    else:
        a = np.zeros((1,), dtype = np.float32)
        b = np.zeros((1,), dtype = np.float32)
        x = np.array([x])
    for i in range(len(a)):
        if x[i] >= 0.3 and x[i] <= 1.1: # IR
            a[i] = 0.574*x[i]**(1.61)
            b[i] = -0.527*x[i]**(1.61)
        elif x[i] >= 1.1 and x[i] <= 3.3: # NIR, Optical
            y = x[i]-1.82
            a[i] = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
            b[i] = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
        elif x[i] >= 3.3 and x[i] <= 8: # UV
            if x[i] >= 5.9 and x[i] <= 8:
                F_a = -0.04473*(x[i] - 5.9)**2 - 0.009779*(x[i] - 5.9)**3
                F_b = 0.2130*(x[i] - 5.9)**2 + 0.1207*(x[i] - 5.9)**3
            else:
                F_a = 0.0
                F_b = 0.0
            a[i] = 1.752 - 0.316*x[i] - 0.104/((x[i] - 4.67)**2 + 0.341) + F_a
            b[i] = -3.090 + 1.825*x[i] + 1.206/((x[i] - 4.62)**2 + 0.263) + F_b
        elif x[i] >= 8 and x[i] <= 10: # Far-UV
            a[i] = -1.073 - 0.628*(x[i] - 8) + 0.137*(x[i] - 8)**2 - 0.070*(x[i] - 8)**3
            b[i] = 13.670 + 4.257*(x[i] - 8) - 0.420*(x[i] - 8)**2 + 0.374*(x[i] - 8)**3
        else:
            a[i] = 0
            b[i] = 0
    if hasattr(lambda_, "__len__"):
        return a + b/R_v
    else:
        return a[0] + b[0]/R_v
    
# Gnedin 2008
def getLMCExtinction(wavelength):
    """
    Compute the extinction (A_lambda / N_H) for LMC dust model at a given wavelength.
    
    Parameters:
    -----------
    wavelength : float or numpy array
        Wavelength(s) in AA.

    Returns:
    --------
    extinction : float or numpy array
        Extinction (A_lambda / N_H) at the given wavelength(s).
    """
    wavelength = wavelength/10**4 # micron
    sigma0 = 3.0*10**(-22) # cm^2, unit of cross-section as intended
    
    # Parameters for LMC Dust Model (from Table 1)
    params_LMC = [
        (0.046, 90, 90, 2, 2),
        (0.08, 19, 21, 4.5, 4.5),
        (0.22, 0.023, -1.95, 2, 2),
        (9.7, 0.005, -1.95, 2, 2),
        (18, 0.006, -1.8, 2, 2),
        (25, 0.02, 0, 2, 2),
        (0.067, 10, 1.9, 4, 15)
    ]
    
    # Initialize extinction to 0
    extinction = np.zeros_like(wavelength)

    # Loop through each term in the LMC dust model
    for lambda_i, a, b, p, q in params_LMC:
        f = lambda x, a, b, p, q: a / (x**p + x**(-q) + b)
        loverli = wavelength/lambda_i # unitless
        extinction += sigma0 * f(loverli, a, b, p, q)

    return extinction # unit of cross-section

def getGamma(lambda_):
    # lambda_ in Angstrom
    if hasattr(lambda_, "__len__"):
        to_return = np.zeros((len(lambda_),), dtype = np.float32)
        has_len = True
    else:
        to_return = np.zeros((1,), dtype = np.float32)
        lambda_ = np.array([lambda_])
        has_len = False
    for i in range(len(lambda_)):
        if lambda_[i] > 2000:
            to_return[i] = 1.6
        else: # lambda_ is outside the SDSS band
            to_return[i] = 1.35
    if has_len:
        return to_return
    else:
        return to_return[0]

def getAbundances(n_H, rel_abundances, atomic_numbers, Z_target, rel_tol):
    # rel_abundances and atomic_numbers must be an array starting with H, then He etc
    Y = 0.2485+1.7756*Z_target
    X = 1 - Z_target - Y
    rel_abundances[atomic_numbers == 2] = Y/X/4
    abundances = rel_abundances*n_H
    # Determine Z with existing values
    Z_tmp = np.sum(abundances[atomic_numbers > 2] * atomic_numbers[atomic_numbers > 2])/np.sum(abundances * atomic_numbers) 
    # Update abundances and rel_abundances s.t. Z_target is satisfied
    abundances[atomic_numbers > 2] = abundances[atomic_numbers > 2]*Z_target/Z_tmp
    rel_abundances = abundances/n_H
    # Check whether we need to fix N-O relation
    O_H = rel_abundances[atomic_numbers == 8]
    N_H = rel_abundances[atomic_numbers == 7]
    if abs(0.41 * O_H * (10**(-1.6) + 10**(2.33 + np.log10(O_H))) - N_H)/(N_H) < rel_tol: # If only < 5 particles left, return
        return abundances
    else:
        # Enforce N-O relationship
        N_H = 0.41 * O_H * (10**(-1.6) + 10**(2.33 + np.log10(O_H)))
        rel_abundances[atomic_numbers == 7] = N_H
        return getAbundances(n_H, rel_abundances, atomic_numbers, Z_target, rel_tol)
    
def process_star_ptc_cloudy(star_ptc_idx, nb_star_ptcs, star_ptc_xyz, star_ptc_mass, star_ptc_initial_mass, star_ptc_formation_time, star_ptc_metallicity, com, a, b, c, norm_vec, rot_matrix, z_sim, cosmic_time_sim, start_time, params, MASS_UNIT, HUBBLE, cosmic_times_interp, nb_rows, ip, wave, isotropic_dust, anisotropic_dust, Nelson):
    theta, phi, RAD, m_H, tau1, tau2, lambda0, alpha1, alpha2, tage, beta, Z_solar, N_H0, cosmic_time_sim, unit_length, N_grid, NSPEC = params
    count_falls_outside = 0
    gx_sp = np.zeros((nb_rows,NSPEC), dtype = np.float32)
    # Suppress contribution if star particle too close to edge of aperture
    proj_vecs_star = (star_ptc_xyz - com)/(HUBBLE/100) - norm_vec*(np.dot((star_ptc_xyz - com)/(HUBBLE/100), norm_vec)) # physical kpc
    # Translate such that plane passes through origin, which is the case already, but just in case
    d = -(a*proj_vecs_star[0]+b*proj_vecs_star[1]+c*proj_vecs_star[2]) # One could take any point in plane, not only the one with index 0
    proj_vecs_star[0] = proj_vecs_star[0] - a * d
    proj_vecs_star[1] = proj_vecs_star[1] - b * d
    proj_vecs_star[2] = proj_vecs_star[2] - c * d
    # Rotate vectors in plane onto plane parallel to xy plane (z=const=0 plane)
    r1 = R.from_matrix(rot_matrix)
    proj_xy_star = r1.apply(proj_vecs_star) # z-coordinate should be const=0
    # Discard if star particle outside Petrosian aperture
    if np.sqrt(proj_xy_star[0]**2+proj_xy_star[1]**2) > RAD:
        count_falls_outside = count_falls_outside + 1
        return gx_sp, wave, star_ptc_mass, count_falls_outside
    else:
        # For seeing effect: Need to determine overlap of Gaussian (FWHM = 7.5 pkpc) with 2D aperture, to make it more reliable
        suppression_fac = 1.0
        proj_xy_star = proj_xy_star + RAD # physical kpc
        
        # Determine age of SSP in Gyr
        cosmic_time_star_ptc = cosmic_times_interp(star_ptc_formation_time)
        age_of_ssp = (cosmic_time_sim-cosmic_time_star_ptc)/1000 # Gyr
        if age_of_ssp <= 0.0001:
            age_of_ssp = 0.0001
        
        # Estimate spectrum via interpolation
        p = [star_ptc_metallicity, age_of_ssp]
        spec = ip([p])[0] # L_sun/Hz
        
        # Correct for actual mass of star particle, and the suppression factor if too close to edge
        spec = spec*(star_ptc_initial_mass*MASS_UNIT/(HUBBLE/100))/1.0/suppression_fac # L_sun/Hz
        
        if star_ptc_idx % 100 == 0:
            print_status(rank, start_time, "Working on star_ptc_idx {}".format(star_ptc_idx))
        if age_of_ssp*1000 < 10: # if younger than 10 Myr, run Cloudy, will overwrite spec and wave
            print_status(rank, start_time, "Found a young stellar particle: star_ptc_idx {} with star_ptc_metallicity {:.3e}, age_of_ssp {:.3f} Myr and stellar mass {:.2e} M_sun".format(star_ptc_idx, star_ptc_metallicity, age_of_ssp*1000, star_ptc_initial_mass*MASS_UNIT/(HUBBLE/100)))
            # Constants
            h_planck = 6.62607015*10**(-34) # J/Hz
            lsun = 3.846e33*10**(-7)  # J/s
            nu0 = 13.6 # eV, Lyman continuum
            nu0 = nu0*h_planck**(-1)*1.60218e-19 # Hz, conversion factor is h**(-1)*1.60218e-19 = 2.418e14, check e.g. Hazy 2
            lightspeed = 299792458 # m/s
            lightspeed = (0.1*10**(-9)/lightspeed)**(-1) # in AA/s
            
            # Cloudy parameters
            Z_ref = 0.02
            age_ref = 1/1000 # Gyr
            U_Sref = 10**(-2) # unitless reference ionization parameter
            cloudy_executable_path = "/data/highz3/cloudy/c23.01/source/cloudy.exe"
            input_file = "sim_{}.in".format(star_ptc_idx)
            Ri_cm = 10**(17) # cm
            n_H = 10**(2.5) # cm**(-3)
            rel_tol = 1e-4
            num_points = 2000 # number of incident radiation field samples
            rel_abundances = 10**(np.array([0.0, -1.01, -10.99, -10.63, -9.47, -3.53, -4.32, -3.17, -7.47, -4.01, -5.70, -4.45, -5.56, -4.48, -6.57, -4.87, -6.53, -5.63, -6.92, -5.67, -8.86, -7.01, -8.03, -6.36, -6.64, -4.51, -7.11, -5.78, -7.82, -7.43])) # at Z_ISM = Z_solar, from Gutkin et al 2016
            atomic_numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
            
            # Calculate abundances
            abundances = getAbundances(n_H, rel_abundances, atomic_numbers, star_ptc_metallicity, rel_tol)
            logrel_abundances = np.log10(abundances/n_H)
            logrel_abundances_str = " ".join([f"{element}={logrel_abundance:.2f}" for element, logrel_abundance in zip(["he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg", "al", "si", "p", "s", "cl", "ar", "k", "ca", "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn"], logrel_abundances[1:])])
            
            # Get number of ionizing photons emitted by the source per second, i.e. Q, from Garg2022BPT
            def integrandQ(nu, Lnu, h_planck):
                # Lnu in units of J
                # h in units of J/Hz
                # nu in units of Hz
                return Lnu(nu)/(h_planck*nu) # unitless
            spec_SI = spec*lsun # J (technically J s^-1 Hz^-1)
            wave_SI = (lightspeed / wave) # Hz
            spec_int = interp1d(wave_SI, spec_SI, kind = 'linear', bounds_error = False, fill_value="extrapolate")
            Q = quad(integrandQ, nu0, np.infty, args=(spec_int, h_planck))[0] # photons per second, already includes M_star factor (for stellar particle)
            
            # Reference Q
            pref = [Z_ref, age_ref]
            spec_ref = ip([pref])[0] # L_sun/Hz
            spec_ref_SI = spec_ref*lsun*(star_ptc_initial_mass*MASS_UNIT/(HUBBLE/100))/1.0/suppression_fac
            spec_int_ref = interp1d(wave_SI, spec_ref_SI, kind = 'linear', bounds_error = False, fill_value="extrapolate")
            Qref = quad(integrandQ, nu0, np.infty, args=(spec_int_ref, h_planck))[0] # photons per second, already includes M_star factor (for stellar particle)
            U_S = U_Sref*(Q/Qref)**(1/3) # unitless ionization parameter
            log10_U_S = np.log10(U_S)
                                    
            # Prepare Cloudy units
            wave_SI = (lightspeed / wave) # Hz
            spec_cloudy = (spec*lsun*10**7)/(4*np.pi*Ri_cm**2) # erg s^-1 Hz^-1 cm^-2, since we need flux density, only used for shape
            
            # Sample the incident radiation field at num_points equidistant points in log10(Hz)
            log10_Hz_values = np.linspace(np.log10(max(wave_SI)), np.log10(min(wave_SI)), num_points) # since wave_cloudy is also monotonically decreasing
            incident_radiation_field = np.interp(log10_Hz_values[::-1], np.log10(wave_SI[::-1]), spec_cloudy[::-1]) # np.interp expects monotonically increasing sample points, # erg s^-1 Hz^-1 cm^-2
                                
            # Prepare the incident radiation field in the required format
            chars_per_line = 70
            interpolate_pairs = ""
            line_length = 0
            line_lengths = []
            for i in range(num_points):
                pair = f"({log10_Hz_values[::-1][i]:.5f} {np.log10(incident_radiation_field[i]):.4f})" # (log nu[Hz], log fnu[erg s^-1 Hz^-1 cm^-2]) pairs, energies must be increasing, but incident_radiation_field was already reversed above
                pair_length = len(pair) + 1  # Add 1 for space
                if line_length + pair_length > chars_per_line:
                    line_lengths.append(line_length)
                    line_length = 0
                interpolate_pairs += pair + " "
                line_length += pair_length

            # Write to sim.in file to prepare Cloudy run
            with open(input_file, "w") as f:
                f.write("title \"this is the input stream for a planetary nebula\"\n")
                f.write("# shape of incident radiation field as ordered pairs giving the energy and intensity\n")
                f.write(f"interpolate {interpolate_pairs[:line_lengths[0]]}\n")
                for line, length in enumerate(line_lengths[1:]):
                    f.write(f"continue {interpolate_pairs[np.sum(line_lengths[:line+1]):np.sum(line_lengths[:line+1])+length]}\n")
                f.write("#\n")
                f.write("# log of starting radius in cm\n")
                f.write(f"radius {np.log10(Ri_cm):.2f}\n")
                f.write("#\n")
                f.write("# log of hydrogen density - cm^-3 -\n")
                f.write(f"hden {np.log10(n_H):.2f}\n")
                f.write("#\n")
                f.write("# this is a sphere with large covering factor, sphere is expanding (photons do not interact with line-absorbing gas on far side)\n")
                f.write("sphere\n")
                f.write("covering factor 1.0\n")
                f.write("#\n")
                f.write("# dimensionless ratio of densities of ionizing photons to hydrogen\n")
                f.write(f"ionization parameter {log10_U_S:.2f}\n")
                f.write("#\n")
                f.write("# abundances taken from Gutkin 2016\n")
                f.write(f"abundances {logrel_abundances_str}\n")
                f.write("#\n")
                f.write("# depletion factors taken from Gutkin 2016 as well\n")
                f.write("metals deplete \"ISM_gutkin.dep\"\n")
                f.write("#\n")
                f.write("# calculation will stop when electron temperature falls below this value\n")
                f.write("stop temperature 4000\n")
                f.write("#\n")
                f.write("# add grains, with abundances scaled to match those assumed for carbon and silicon above, PAHs are important at 10^5 AA only\n")
                f.write("grains Orion\n")
                f.write("#\n")
                f.write("# save output spectrum\n")
                f.write(f"save continuum units Angstroms \"sed_out_{star_ptc_initial_mass:.2e}_{star_ptc_metallicity:.2e}_{age_of_ssp:.2e}.txt\"\n")
            
            # Run Cloudy
            subprocess.run([cloudy_executable_path, input_file]) # raise Python error if process returns a non-zero exit status
            
            # Read in net transmitted continuum (column 5, index 4)
            sed_out = np.loadtxt(f"./sim_{star_ptc_idx}sed_out_{star_ptc_initial_mass:.2e}_{star_ptc_metallicity:.2e}_{age_of_ssp:.2e}.txt", skiprows=1, usecols=[0, 1, 2, 3, 4])
            wave_AA = sed_out[:, 0] # AA
            wave_Hz = (lightspeed / wave_AA) # Hz
            # Renormalize based on incident radiation field & target = spec
            spec_in = sed_out[:, 1]/(lsun*10**7)/wave_Hz # L_sun/Hz
            spec_ini = interp1d(wave_AA[::-1], spec_in[::-1], bounds_error = False, fill_value = 'extrapolate') # Swap around so that wave_AA is monotonically increasing
            spec_i = spec_ini(wave) # L_sun/Hz
            renormalise = np.max(spec)/np.max(spec_i)
            net_trans = sed_out[:, 4] # nu L_nu, erg s^-1, even though it should be nu L_nu/(4\pi r_0^2) # erg s^-1 cm^-2 (attenuated incident + nebular continua + lines)
            # If you use intensity command then: 4\pi nu J_nu/(4\pi r_0^2), erg s^-1, even though it should be 4\pi nu J_nu # erg s^-1 cm^-2 
            spec_out = net_trans/(lsun*10**7)/wave_Hz*renormalise # L_sun/Hz
            # Interpolate at old wave values
            spec_outi = interp1d(wave_AA[::-1], spec_out[::-1], bounds_error = False, fill_value = 'extrapolate') # Swap around so that wave_AA is monotonically increasing
            spec = spec_outi(wave)
            # Delete nuisance files
            os.remove(f"./sim_{star_ptc_idx}sed_out_{star_ptc_initial_mass:.2e}_{star_ptc_metallicity:.2e}_{age_of_ssp:.2e}.txt")
            os.remove(f"./sim_{star_ptc_idx}.in")
            os.remove(f"./sim_{star_ptc_idx}.out")
            
        gx_sp[0] += spec
        if isotropic_dust == True:
            # Attenuated spectra, isotropic dust effect
            tau_values = np.zeros((len(wave),), dtype = np.float32)
            for i in range(len(wave)):
                if age_of_ssp*1000 <= tage:
                    tau_values[i] = tau1*(wave[i]/lambda0)**(-alpha1)
                else:
                    tau_values[i] = tau2*(wave[i]/lambda0)**(-alpha2)
            spec_attenuated = spec*np.exp(-tau_values)
            gx_sp[1] += spec_attenuated
            if anisotropic_dust == True:
                DELTA_X = 2*RAD/N_grid
                # Model C in Nelson 2018, find bilinear interpolation of mass-weighted Z grid to position of star particle
                x = np.linspace(0, 2*RAD-DELTA_X, N_grid) + DELTA_X/2 # physical kpc
                y = np.linspace(0, 2*RAD-DELTA_X, N_grid) + DELTA_X/2
                interp = RegularGridInterpolator((x, y), config.metallicity_grid, bounds_error=False, fill_value=None)
                Z_g = interp(proj_xy_star[:2])
                interp = RegularGridInterpolator((x, y), config.col_dens_grid, bounds_error=False, fill_value=None)
                N_H = interp(proj_xy_star[:2])
                if Nelson:
                    tau_lambda_a = (getSolarExtinction(wave))*(1+z_sim)**(beta)*(Z_g/Z_solar)**getGamma(wave)*(N_H/N_H0)
                else:
                    # Note that actually you should be doing \sum Z n_HI + f_ion \sum Z n_HII, i.e. factor in Z on a cell basis, not project n_HI and n_HII first as you do now, you lose information
                    interp = RegularGridInterpolator((x, y), config.colHII_dens_grid, bounds_error=False, fill_value=None)
                    N_HII = interp(proj_xy_star[:2])
                    f_ion2dust = 0.01  # From Laursen+2009
                    dust2metal_ratio = 0.4  # From Kaviraj+2017, actually Draine, see also Sergio's SOFIA paper, https://arxiv.org/pdf/2311.06356
                    tau_lambda_a = dust2metal_ratio*getLMCExtinction(wave)*np.log(10)/2.5*(N_H + f_ion2dust*N_HII)/Z_solar
                tau_lambda = tau_lambda_a*(getAnisoScatt(wave)*(1-getAlbedo(wave))**(1/2)+(1-getAnisoScatt(wave))*(1-getAlbedo(wave)))
                nelson_att_factor = np.zeros((wave.shape[0],), dtype = np.float32)
                for i in range(len(wave)):
                    if tau_lambda[i] != 0.0:
                        nelson_att_factor[i] = 1/tau_lambda[i]*(1-np.exp(-tau_lambda[i]))
                    else:
                        nelson_att_factor[i] = 1
                if star_ptc_idx % 100 == 0:
                    if Nelson:
                        print_status(rank, start_time, "star_ptc_idx {}, nelson_att_factor is {}, getGamma(wave) is {}, getSolarExtinction(wave) is {}, (getAnisoScatt(wave)*(1-getAlbedo(wave))**(1/2)+(1-getAnisoScatt(wave))*(1-getAlbedo(wave))) is {}, Z_g/Z_solar is {}, N_H/N_H0 is {}".format(star_ptc_idx, nelson_att_factor[np.logical_and(wave>10**3, wave<10**4)], getGamma(wave)[np.logical_and(wave>10**3, wave<10**4)], getSolarExtinction(wave)[np.logical_and(wave>10**3, wave<10**4)], (getAnisoScatt(wave)*(1-getAlbedo(wave))**(1/2)+(1-getAnisoScatt(wave))*(1-getAlbedo(wave)))[np.logical_and(wave>10**3, wave<10**4)], Z_g/Z_solar, N_H/N_H0))
                    else:
                        print_status(rank, start_time, "star_ptc_idx {}, att_factor is {}, getLMCExtinction(wave) is {}, (getAnisoScatt(wave)*(1-getAlbedo(wave))**(1/2)+(1-getAnisoScatt(wave))*(1-getAlbedo(wave))) is {}, N_HI*Z column is {}, N_HII*Z column is {}".format(star_ptc_idx, nelson_att_factor[np.logical_and(wave>10**3, wave<10**4)], getLMCExtinction(wave)[np.logical_and(wave>10**3, wave<10**4)], (getAnisoScatt(wave)*(1-getAlbedo(wave))**(1/2)+(1-getAnisoScatt(wave))*(1-getAlbedo(wave)))[np.logical_and(wave>10**3, wave<10**4)], N_H, N_HII))
                gx_sp[2] += spec_attenuated*nelson_att_factor
        return gx_sp, wave, star_ptc_mass, count_falls_outside # star_ptc_mass in code units
    
def process_star_ptc(star_ptc_idx, nb_star_ptcs, star_ptc_xyz, star_ptc_mass, star_ptc_initial_mass, star_ptc_formation_time, star_ptc_metallicity, com, a, b, c, norm_vec, rot_matrix, z_sim, cosmic_time_sim, start_time, params, MASS_UNIT, HUBBLE, cosmic_times_interp, nb_rows, ip, wave, isotropic_dust, anisotropic_dust, Nelson):
    theta, phi, RAD, m_H, tau1, tau2, lambda0, alpha1, alpha2, tage, beta, Z_solar, N_H0, cosmic_time_sim, unit_length, N_grid, NSPEC = params
    count_falls_outside = 0
    gx_sp = np.zeros((nb_rows,NSPEC), dtype = np.float32)
    # Suppress contribution if star particle too close to edge of aperture
    proj_vecs_star = (star_ptc_xyz - com)/(HUBBLE/100) - norm_vec*(np.dot((star_ptc_xyz - com)/(HUBBLE/100), norm_vec)) # physical kpc
    # Translate such that plane passes through origin, which is the case already, but just in case
    d = -(a*proj_vecs_star[0]+b*proj_vecs_star[1]+c*proj_vecs_star[2]) # One could take any point in plane, not only the one with index 0
    proj_vecs_star[0] = proj_vecs_star[0] - a * d
    proj_vecs_star[1] = proj_vecs_star[1] - b * d
    proj_vecs_star[2] = proj_vecs_star[2] - c * d
    # Rotate vectors in plane onto plane parallel to xy plane (z=const=0 plane)
    r1 = R.from_matrix(rot_matrix)
    proj_xy_star = r1.apply(proj_vecs_star) # z-coordinate should be const=0
    # Discard if star particle outside Petrosian aperture
    if np.sqrt(proj_xy_star[0]**2+proj_xy_star[1]**2) > RAD:
        count_falls_outside = count_falls_outside + 1
        return gx_sp, wave, star_ptc_mass, count_falls_outside
    else:
        # For seeing effect: Need to determine overlap of Gaussian (FWHM = 7.5 pkpc) with 2D aperture, to make it more reliable
        suppression_fac = 1.0
        proj_xy_star = proj_xy_star + RAD # physical kpc
        
        # Determine age of SSP in Gyr
        cosmic_time_star_ptc = cosmic_times_interp(star_ptc_formation_time)
        age_of_ssp = (cosmic_time_sim-cosmic_time_star_ptc)/1000 # Gyr
        if age_of_ssp <= 0.0001:
            age_of_ssp = 0.0001
        
        # Estimate spectrum via interpolation
        p = [star_ptc_metallicity, age_of_ssp]
        spec = ip([p])[0]
        
        # Correct for actual mass of star particle, and the suppression factor if too close to edge
        spec = spec*(star_ptc_initial_mass*MASS_UNIT/(HUBBLE/100))/1.0/suppression_fac # sp.stellar_mass is surviving solar mass
        
        gx_sp[0] += spec
        if isotropic_dust == True:
            # Attenuated spectra, isotropic dust effect
            tau_values = np.zeros((len(wave),), dtype = np.float32)
            for i in range(len(wave)):
                if age_of_ssp*1000 <= tage:
                    tau_values[i] = tau1*(wave[i]/lambda0)**(-alpha1)
                else:
                    tau_values[i] = tau2*(wave[i]/lambda0)**(-alpha2)
            spec_attenuated = spec*np.exp(-tau_values)
            gx_sp[1] += spec_attenuated
            if anisotropic_dust == True:
                DELTA_X = 2*RAD/N_grid
                # Model C in Nelson 2018, find bilinear interpolation of mass-weighted Z grid to position of star particle
                x = np.linspace(0, 2*RAD-DELTA_X, N_grid) + DELTA_X/2 # physical kpc
                y = np.linspace(0, 2*RAD-DELTA_X, N_grid) + DELTA_X/2
                interp = RegularGridInterpolator((x, y), config.metallicity_grid, bounds_error=False, fill_value=None)
                Z_g = interp(proj_xy_star[:2])
                interp = RegularGridInterpolator((x, y), config.col_dens_grid, bounds_error=False, fill_value=None)
                N_H = interp(proj_xy_star[:2])
                if Nelson:
                    tau_lambda_a = (getSolarExtinction(wave))*(1+z_sim)**(beta)*(Z_g/Z_solar)**getGamma(wave)*(N_H/N_H0)
                else:
                    # Note that actually you should be doing \sum Z n_HI + f_ion \sum Z n_HII, i.e. factor in Z on a cell basis, not project n_HI and n_HII first as you do now, you lose information
                    interp = RegularGridInterpolator((x, y), config.colHII_dens_grid, bounds_error=False, fill_value=None)
                    N_HII = interp(proj_xy_star[:2])
                    f_ion2dust = 0.01  # From Laursen+2009
                    dust2metal_ratio = 0.4  # From Kaviraj+2017, actually Draine, see also Sergio's SOFIA paper, https://arxiv.org/pdf/2311.06356
                    tau_lambda_a = dust2metal_ratio*getLMCExtinction(wave)*np.log(10)/2.5*(N_H + f_ion2dust*N_HII)/Z_solar
                tau_lambda = tau_lambda_a*(getAnisoScatt(wave)*(1-getAlbedo(wave))**(1/2)+(1-getAnisoScatt(wave))*(1-getAlbedo(wave)))
                nelson_att_factor = np.zeros((wave.shape[0],), dtype = np.float32)
                for i in range(len(wave)):
                    if tau_lambda[i] != 0.0:
                        nelson_att_factor[i] = 1/tau_lambda[i]*(1-np.exp(-tau_lambda[i]))
                    else:
                        nelson_att_factor[i] = 1
                if star_ptc_idx % 100 == 0:
                    if Nelson:
                        print_status(rank, start_time, "star_ptc_idx {}, nelson_att_factor is {}, getGamma(wave) is {}, getSolarExtinction(wave) is {}, (getAnisoScatt(wave)*(1-getAlbedo(wave))**(1/2)+(1-getAnisoScatt(wave))*(1-getAlbedo(wave))) is {}, Z_g/Z_solar is {}, N_H/N_H0 is {}".format(star_ptc_idx, nelson_att_factor[np.logical_and(wave>10**3, wave<10**4)], getGamma(wave)[np.logical_and(wave>10**3, wave<10**4)], getSolarExtinction(wave)[np.logical_and(wave>10**3, wave<10**4)], (getAnisoScatt(wave)*(1-getAlbedo(wave))**(1/2)+(1-getAnisoScatt(wave))*(1-getAlbedo(wave)))[np.logical_and(wave>10**3, wave<10**4)], Z_g/Z_solar, N_H/N_H0))
                    else:
                        print_status(rank, start_time, "star_ptc_idx {}, att_factor is {}, getLMCExtinction(wave) is {}, (getAnisoScatt(wave)*(1-getAlbedo(wave))**(1/2)+(1-getAnisoScatt(wave))*(1-getAlbedo(wave))) is {}, N_HI*Z column is {}, N_HII*Z column is {}".format(star_ptc_idx, nelson_att_factor[np.logical_and(wave>10**3, wave<10**4)], getLMCExtinction(wave)[np.logical_and(wave>10**3, wave<10**4)], (getAnisoScatt(wave)*(1-getAlbedo(wave))**(1/2)+(1-getAnisoScatt(wave))*(1-getAlbedo(wave)))[np.logical_and(wave>10**3, wave<10**4)], N_H, N_HII))
                gx_sp[2] += spec_attenuated*nelson_att_factor
        return gx_sp, wave, star_ptc_mass, count_falls_outside # star_ptc_mass in code units
        
def respectPBCNoRef(xyz, L_BOX):
    """
    Return modified positions xyz_out of an object that respect the box periodicity
    
    If point distro xyz has particles separated in any Cartesian direction
    by more than config.L_BOX/2, translate those particles accordingly.
    
    :param xyz: coordinates of particles
    :type xyz: (N^3x3) floats
    :return: updated coordinates of particles
    :rtype: (N^3x3) floats"""
    xyz_out = xyz.copy() # Otherwise changes would be reflected in outer scope (np.array is mutable).
    ref = 0 # Reference particle does not matter
    dist_x = xyz_out[:,0]-xyz_out[ref, 0]
    dist_y = xyz_out[:,1]-xyz_out[ref, 1]
    dist_z = xyz_out[:,2]-xyz_out[ref, 2]
    xyz_out[:,0][dist_x > L_BOX/2] = xyz_out[:,0][dist_x > L_BOX/2]-L_BOX
    xyz_out[:,0][dist_x < -L_BOX/2] = xyz_out[:,0][dist_x < -L_BOX/2]+L_BOX
    xyz_out[:,1][dist_y > L_BOX/2] = xyz_out[:,1][dist_y > L_BOX/2]-L_BOX
    xyz_out[:,1][dist_y < -L_BOX/2] = xyz_out[:,1][dist_y < -L_BOX/2]+L_BOX
    xyz_out[:,2][dist_z > L_BOX/2] = xyz_out[:,2][dist_z > L_BOX/2]-L_BOX
    xyz_out[:,2][dist_z < -L_BOX/2] = xyz_out[:,2][dist_z < -L_BOX/2]+L_BOX
    return xyz_out

def calcMode(xyz, masses, rad):
    """ Find mode (point of highest local density) of point distribution xyz

    :param xyz: coordinates of particles of type 1 or type 4
    :type xyz: (N,3) floats
    :param masses: masses of the particles
    :type masses: (N,) floats
    :param rad: initial radius to consider from CoM of object
    :type rad: float
    :return: mode of distro
    :rtype: (3,) floats"""
    com = np.sum(xyz*np.reshape(masses, (masses.shape[0],1)), axis = 0, dtype=np.float64)/masses.sum()
    distances_all = np.linalg.norm(xyz-com,axis=1)
    xyz_constrain = xyz[distances_all < rad]
    masses_constrain = masses[distances_all < rad]
    if xyz_constrain.shape[0] < 5: # If only < 5 particles left, return
        return com
    else:
        rad *= 0.83 # Reduce radius by 17 %
        return calcMode(xyz_constrain, masses_constrain, rad)

def respectPBCPtRef(xyz, pt, L_BOX):
    """
    Return modified positions xyz_out of an object that respect the box periodicity
    
    If point distro xyz has particles separated in any Cartesian direction
    by more than config.L_BOX/2, translate those particles accordingly.
    
    :param xyz: coordinates of particles
    :type xyz: (N^3x3) floats
    :return: updated coordinates of particles
    :rtype: (N^3x3) floats"""
    xyz_out = xyz.copy() # Otherwise changes would be reflected in outer scope (np.array is mutable).
    dist_x = xyz_out[:,0]-pt[0]
    dist_y = xyz_out[:,1]-pt[1]
    dist_z = xyz_out[:,2]-pt[2]
    xyz_out[:,0][dist_x > L_BOX/2] = xyz_out[:,0][dist_x > L_BOX/2]-L_BOX
    xyz_out[:,0][dist_x < -L_BOX/2] = xyz_out[:,0][dist_x < -L_BOX/2]+L_BOX
    xyz_out[:,1][dist_y > L_BOX/2] = xyz_out[:,1][dist_y > L_BOX/2]-L_BOX
    xyz_out[:,1][dist_y < -L_BOX/2] = xyz_out[:,1][dist_y < -L_BOX/2]+L_BOX
    xyz_out[:,2][dist_z > L_BOX/2] = xyz_out[:,2][dist_z > L_BOX/2]-L_BOX
    xyz_out[:,2][dist_z < -L_BOX/2] = xyz_out[:,2][dist_z < -L_BOX/2]+L_BOX
    return xyz_out

def gx_spectrum_parallel(start_time, gas_xyz, gas_masses, V, metallicity_gas, m_HI_in_grams, m_HII_in_grams, star_xyz, star_initial_masses, star_masses, metallicity, star_formation_time, cosmic_times_interp, params, z_sim, isotropic_dust, anisotropic_dust, ip, HUBBLE, L_BOX, wave, plot_params, cloudy = False, Nelson = True):
    """ Calculate spectrum of 1 galaxy, parallelized
    :param start_time: for printing purposes
    :type start_time: float
    :param gas_xyz: gas cell positions in units of pkpc/h
    :type gas_xyz: (N,3) floats
    :param gas_masses: gas cell masses in code units (10^10 M_sun/h)
    :type gas_masses: (N,) floats
    :param V: cell volume in units of (pkpc/h)**3
    :type V: (N,) floats
    :param metallicity_gas: metallicity mass of gas cell, pure mass in metals (10^10 M_sun/h)
    :type metallicity_gas: (N,) floats
    :param m_HI_in_grams: HI mass in gas cell in g
    :type m_HI_in_grams: (N,) floats
    :param m_HII_in_grams: HII mass in gas cell in g
    :type m_HII_in_grams: (N,) floats
    :param star_xyz: star particle positions in units of pkpc/h
    :type star_xyz: (N1,3) floats
    :param star_initial_masses: star particle masses at formation time in 10^10 M_sun/h
    :type star_initial_masses: (N1,) floats
    :param star_masses: star particle masses in 10^10 M_sun/h
    :type star_masses: (N1,) floats
    :param metallicity: metallicity of star particle, mass fraction (not /Z_0)
    :type metallicity: (N1,) floats
    :param star_formation_time: scale factors when star particles formed, unitless
    :type star_formation_time: (N1,) floats
    :param cosmic_times_interp: interpolator object, a --> cosmic time
    :type cosmic_times_interp: callable
    :param params: various numerical and physical params for the model
    :type params: tuple
    :param z_sim: redshift of galaxy, unitless
    :type z_sim: float
    :param isotropic_dust: boolean for activating / deactivating dust attenuation
    :type isotropic_dust: boolean
    :param anisotropic_dust: boolean for activating / deactivating Nelson C dust attenuation model
    :type anisotropic_dust: boolean
    :param ip: interpolator object for spectra
    :type ip: interpolator
    :param HUBBLE: Hubble constant in km/s/Mpc
    :type HUBBLE: float
    :param L_BOX: box size in pkpc/h
    :type L_BOX: float
    :param wave: wavelengths in AA
    :type wave: (5994,) floats
    :return: spectrum of galaxy, in units of L_sun / Hz
    :rtype: (1,NSPEC), (2,NSPEC) or (3,NSPEC) (if both isotropic_dust and anisotropic_dust are True)
    """
    # Unpack parameters, numerical and physical
    VIZ_DEST, angle, sim = plot_params
    theta, phi, RAD, m_H, tau1, tau2, lambda0, alpha1, alpha2, tage, beta, Z_solar, N_H0, cosmic_time_sim, unit_length, N_grid, NSPEC = params
    # Define spectrum, size depends on whether isotropic_dust, anisotropic_dust are activated
    if isotropic_dust == False and anisotropic_dust == False:
        nb_rows = 1
    elif isotropic_dust and anisotropic_dust == False:
        nb_rows = 2
    elif isotropic_dust and anisotropic_dust:
        nb_rows = 3
    else:
        raise ValueError("You cannot have isotropic_dust == False but anisotropic_dust == True")
    # COM
    gas_xyz = respectPBCNoRef(gas_xyz, L_BOX) # in pkpc/h
    com = calcMode(gas_xyz, gas_masses, rad=1000) # rad in pkpc/h
    print_status(rank, start_time, "Mode of gx is {} pkpc/h".format(com))
    star_xyz = respectPBCPtRef(star_xyz, com, L_BOX)
    
    # Define normal vector orthographic projection grid
    norm_vec = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    a = norm_vec[0]
    b = norm_vec[1]
    c = norm_vec[2]
    denom = np.sqrt(a**2+b**2+c**2) # Should be 1 anyway
    cost = c/denom
    sint = np.sqrt(a**2+b**2)/denom
    u1 = b/np.sqrt(a**2+b**2)
    u2 = -a/np.sqrt(a**2+b**2)
    rot_matrix = np.array([[cost+u1**2*(1-cost), u1*u2*(1-cost), u2*sint],
                       [u1*u2*(1-cost), cost+u2**2*(1-cost), -u1*sint],
                       [-u2*sint, u1*sint, cost]])
    r1 = R.from_matrix(rot_matrix)
    to_add = np.arange(len(star_xyz))
    if anisotropic_dust:
        proj_vecs = np.zeros((len(gas_xyz), 3), dtype = np.float32)
        for i, xyz in enumerate(gas_xyz):
            proj_vecs[i] = (xyz - com)/(HUBBLE/100) - norm_vec*(np.dot((xyz-com)/(HUBBLE/100), norm_vec)) # physical kpc
        # Translate such that plane passes through origin, which is the case already by the way, but just in case
        d = -(a*proj_vecs[0,0]+b*proj_vecs[0,1]+c*proj_vecs[0,2]) # Distance from origin to the plane. One could take any point in plane, not only the one with index 0
        proj_vecs[:, 0] = proj_vecs[:, 0] - a * d
        proj_vecs[:, 1] = proj_vecs[:, 1] - b * d
        proj_vecs[:, 2] = proj_vecs[:, 2] - c * d
        # Rotate vectors in plane onto plane parallel to xy plane (z=const=0 plane)
        proj_xy = r1.apply(proj_vecs) # z-coordinates should be const=0
        # Translation in plane
        proj_xy = proj_xy + RAD # center should become left-bottom edge of grid, physical kpc
        # Discard gas cells outside of enlarged, square-shaped "aperture"
        keep_x = np.logical_and(proj_xy[:,0] >= 0.0, proj_xy[:,0] < 2*RAD)
        keep_y = np.logical_and(proj_xy[:,1] >= 0.0, proj_xy[:,1] < 2*RAD)
        keep = np.logical_and(keep_x, keep_y)
        proj_xy = proj_xy[keep] # already pkpc, so it matches hsml units
        gas_to_add = np.arange(len(gas_xyz))
        gas_to_add = gas_to_add[keep]
        # Prepare cubic spline kernel deposition onto grid
        r_cell = (3*V[gas_to_add]/(4*np.pi)/(HUBBLE/100)**3)**(1/3) # physical kpc
        vol_cell = V[gas_to_add]/(HUBBLE/100)**3 # physical kpc**3
        hsml = np.float32(2.5*r_cell) # physical kpc
        gas_masses[gas_masses < 1e-15] = 1e-15 # 10^10 M_sun/h, to avoid nans after division by zero
        HI_weighted_Z = np.float32(metallicity_gas[gas_to_add]/gas_masses[gas_to_add]*m_HI_in_grams[gas_to_add]/np.mean(m_HI_in_grams[gas_to_add])) # unitless, could add weighting via *m_HI_in_grams[gas_to_add]/np.mean(m_HI_in_grams[gas_to_add])
        metallicity_grid = makeGridWithSPHNoPBC(np.float32(proj_xy[:,0]), np.float32(proj_xy[:,1]), np.float32(HI_weighted_Z), np.float32(hsml), np.float32(RAD*2), N_grid) # 1 / pkpc**2
        metallicity_grid = (1+(metallicity_grid-np.mean(metallicity_grid))/np.mean(metallicity_grid))*np.mean(HI_weighted_Z) # unitless
                
        # Column density
        col_dens_cells = np.float64(1/m_H*2*r_cell*m_HI_in_grams[gas_to_add]/vol_cell) # in pkpc**-2
        col_dens_cells_cmMinus2 = np.float64(col_dens_cells/unit_length**2) # in cm**-2
        colHII_dens_cells = np.float64(1/m_H*2*r_cell*m_HII_in_grams[gas_to_add]/vol_cell) # in pkpc**-2
        colHII_dens_cells_cmMinus2 = np.float64(colHII_dens_cells/unit_length**2) # in cm**-2
        col_dens_grid = makeGridWithSPHNoPBC(np.float32(proj_xy[:,0]), np.float32(proj_xy[:,1]), np.float32(col_dens_cells_cmMinus2), np.float32(hsml), np.float32(RAD*2), N_grid)
        colHII_dens_grid = makeGridWithSPHNoPBC(np.float32(proj_xy[:,0]), np.float32(proj_xy[:,1]), np.float32(colHII_dens_cells_cmMinus2), np.float32(hsml), np.float32(RAD*2), N_grid)
        # Fix units of grids
        col_dens_grid = (1+(col_dens_grid-np.mean(col_dens_grid))/np.mean(col_dens_grid))*np.mean(col_dens_cells_cmMinus2) # in cm**-2
        colHII_dens_grid = (1+(colHII_dens_grid-np.mean(colHII_dens_grid))/np.mean(colHII_dens_grid))*np.mean(colHII_dens_cells_cmMinus2) # in cm**-2
    
    # Make grids available globally
    print_status(rank, start_time, "m_HI_in_grams.mean is {:.2e} m_HI_in_grams.max is {:.2e} and m_HI_in_grams.min is {:.2e}".format(m_HI_in_grams.mean(), m_HI_in_grams.max(), m_HI_in_grams.min()))
    if anisotropic_dust:
        print_status(rank, start_time, "vol_cell.mean is {:.2e} vol_cell.max is {:.2e} and vol_cell.min is {:.2e}".format(vol_cell.mean(), vol_cell.max(), vol_cell.min()))
        print_status(rank, start_time, "hsml.mean is {:.2e} hsml.max is {:.2e} and hsml.min is {:.2e}".format(hsml.mean(), hsml.max(), hsml.min()))
        print_status(rank, start_time, "col_dens_cells_cmMinus2.mean is {:.2e} col_dens_cells_cmMinus2.max is {:.2e} and col_dens_cells_cmMinus2.min is {:.2e}".format(col_dens_cells_cmMinus2.mean(), col_dens_cells_cmMinus2.max(), col_dens_cells_cmMinus2.min()))
        print_status(rank, start_time, "col_dens_grid.mean is {:.2e} col_dens_grid.max is {:.2e} and col_dens_grid.min is {:.2e}".format(col_dens_grid.mean(), col_dens_grid.max(), col_dens_grid.min()))
        print_status(rank, start_time, "colHII_dens_grid.mean is {:.2e} colHII_dens_grid.max is {:.2e} and colHII_dens_grid.min is {:.2e}".format(colHII_dens_grid.mean(), colHII_dens_grid.max(), colHII_dens_grid.min()))
        print_status(rank, start_time, "metallicity_grid.mean is {:.2e} metallicity_grid.max is {:.2e} and metallicity_grid.min is {:.2e}".format(metallicity_grid.mean(), metallicity_grid.max(), metallicity_grid.min()))
        print_status(rank, start_time, "HI_weighted_Z.mean is {:.2e} HI_weighted_Z.max is {:.2e} and HI_weighted_Z.min is {:.2e}".format(HI_weighted_Z.mean(), HI_weighted_Z.max(), HI_weighted_Z.min()))
        updateZColGrids(metallicity_grid, col_dens_grid, colHII_dens_grid)
        # Plot Zgas and NHI grids
        vizZAndHI(start_time, RAD, Z_solar, VIZ_DEST, angle, sim, z_sim)
    
    # Define stellar population object
    gx_sp = np.zeros((nb_rows,NSPEC), dtype = np.float32)
    total_stellar_mass = 0.0
    count_falls_outside = 0
    MASS_UNIT = 10**10
    with mp.Pool() as pool:
        if cloudy:
            results = pool.starmap(process_star_ptc_cloudy, [(star_ptc_idx, len(to_add), star_xyz[star_ptc_idx], star_masses[star_ptc_idx], star_initial_masses[star_ptc_idx], star_formation_time[star_ptc_idx], metallicity[star_ptc_idx], com, a, b, c, norm_vec, rot_matrix, z_sim, cosmic_time_sim, start_time, params, MASS_UNIT, HUBBLE, cosmic_times_interp, nb_rows, ip, wave, isotropic_dust, anisotropic_dust, Nelson) for star_ptc_idx in range(len(to_add))])
        else:
            results = pool.starmap(process_star_ptc, [(star_ptc_idx, len(to_add), star_xyz[star_ptc_idx], star_masses[star_ptc_idx], star_initial_masses[star_ptc_idx], star_formation_time[star_ptc_idx], metallicity[star_ptc_idx], com, a, b, c, norm_vec, rot_matrix, z_sim, cosmic_time_sim, start_time, params, MASS_UNIT, HUBBLE, cosmic_times_interp, nb_rows, ip, wave, isotropic_dust, anisotropic_dust, Nelson) for star_ptc_idx in range(len(to_add))])
        for result in results:
            if result[0][0, NSPEC//2] != 0.0:
                gx_sp += result[0]
                wave = result[1]
                total_stellar_mass += result[2]
            else:
                wave = result[1]
            count_falls_outside += result[3]
    print_status(rank, start_time, "In total, among {} star particles, {} have fallen outside of (typically twice) factor x stellar half-mass radius. Note that this is just a consistency check.".format(len(to_add), count_falls_outside))
    
    return gx_sp, wave, total_stellar_mass
