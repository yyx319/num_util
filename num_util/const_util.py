# constant util
# value of constants
# unit conversion
import numpy as np 
import astropy.units as u; import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM

# constants
m_p = c.m_p.cgs.value
c_light = c.c.cgs.value
kb      = c.k_B.cgs.value       # [erg/K] Boltzman constant
amu     = 1.67377e-24         # [g] atomic mass unit

# unit conversion
Sr2arcsec2 = (u.sr).to(u.arcsec**2)
pc2cm  = (u.pc).to(u.cm)
kpc2cm = (u.kpc).to(u.cm)
Mpc2cm = (u.Mpc).to(u.cm)
cm2pc  = (u.cm).to(u.pc)
cm2kpc = (u.cm).to(u.kpc)
cm2Mpc = (u.cm).to(u.Mpc)

s2yr  = (u.s).to(u.yr)
s2Myr = (u.s).to(u.Myr)
s2Gyr = (u.s).to(u.Gyr)
yr2s  = (u.yr).to(u.s)
Myr2s = (u.Myr).to(u.s)
Gyr2s = (u.Gyr).to(u.s)

g2Msun = (u.g).to(u.M_sun)
Msun2g = (u.Msun).to(u.g)

# cosmological parameters
h0 = 0.6790000152587891*100
tH = 1./(h0*u.km/u.Mpc/u.s)
tH = tH.to(u.Gyr).value 
Om0 = 0.306499987840652
Tcmb0 = 2.725
cosmo = FlatLambdaCDM(H0=h0, Om0=Om0, Tcmb0=Tcmb0)

f_H = 0.76
f_He = 0.24

Z_sol = 0.0134 # Asplund09 Table 4

# line parameters
lam_lya = 1215.67    
lam_D_lya = 1215.34

def eV2Angstrom(E):
    lam = c.h*c.c/(E*u.eV)
    lam = lam.to(u.Angstrom).value
    return lam

def Angstrom2eV(lam):
    E = c.h*c.c/(lam*u.Angstrom)
    E = E.to(u.eV).value
    return E



