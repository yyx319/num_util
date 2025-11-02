import numpy as np 
import matplotlib.pyplot as plt 
import astropy.units as u; import astropy.constants as c

def cal_beta(uth, coolrate, Mbin, R):
    ''' 
    function to calculate beta cooling parameter 
    uth: specific internal energy
    coolrate: 
    '''
    
    tcool = uth/coolrate 
    Omega_K = (c.G*Mbin/R**3)**(1/2)
    
    beta=Omega_K*tcool 
    beta=beta.decompose()
    return beta