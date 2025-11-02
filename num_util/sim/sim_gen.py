from cProfile import label
from lib2to3.pygram import python_grammar_no_print_statement
import re
import os
import sys
import warnings

import numpy as np 
import random
import matplotlib.pyplot as plt
import scipy 
from scipy.ndimage import gaussian_filter
import astropy.units as u; import astropy.constants as c
from scipy import stats



def op_idx_int2str(op_idx, L=5):
    '''
    covert op_idx from int to string
    Input: 
        op_idx: output index. int type
        L: total length of string
    Output:
        op_idx: output index. str type
    ex: op_idx=55 L=5 -> '00055'
    '''
    if type(op_idx)==int or type(op_idx)==np.int64:
        pass
    else:
        raise Exception('error, input op_idx is not int')
    op_idx = str(op_idx)
    if len(op_idx)<L:
        op_idx = '0'*(L-len(op_idx)) + op_idx
    elif len(op_idx)==L:
        pass
    elif len(op_idx)>5:
        raise Exception('error, output index is longer than the largest length')
    return op_idx


def op_idx_checkstr(op_idx, L=5):
    '''
    check if op_idx follow the str format, if not change it to required format
    '''
    if type(op_idx)!=str:
        raise Exception('op_idx is not a string')
    elif type(op_idx)==str:
        pass

    if len(op_idx)<L:
        op_idx = '0'*( L-len(op_idx) ) + op_idx
    elif len(op_idx)==L:
        pass
    elif len(op_idx)>L:
        raise Exception('error, output index is longer than the largest length')
    
    return op_idx