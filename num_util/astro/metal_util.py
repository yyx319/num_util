import numpy as np
import matplotlib.pyplot as plt
from molmass import Formula

from num_util.const_util import *


def Z_to_12plogOH(Z):
    '''
    assume oxygen is 35 percent of metal mass
    '''
    f_mass_OM=0.35  # Torrey+19 section 2.2
    m_atom_O = Formula('O').mass
    m_atom_H = Formula('H').mass
    
    ObyH = f_mass_OM*Z/( f_H*m_atom_O/m_atom_H )
    
    return 12+np.log10(ObyH)