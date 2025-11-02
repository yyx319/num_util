'''
machine learning util
'''
import numpy as np
import math 
from astropy.modeling.models import Sersic1D
import matplotlib.pyplot as plt


import num_util 
from num_util.const_util import *

def clean_X_y(X,y):
    '''
    clean up X and y. Remove the index where there is any nan in X or y
    '''
    idx1 = np.where( X!=X )[0]
    idx2 = np.where( y!=y )[0]
    idx = np.concatenate( (idx1, idx2), axis=0 )
    
    idx = np.unique(idx)
    
    X = np.delete(X, idx, axis=0)
    y = np.delete(y, idx)
    return X, y

def IoR():
    '''
    Bluck20
    '''
    return 0

def PCC():
    rho_AB_C = 0
    return 0
    

# linear model

# decision tree

# random forest



