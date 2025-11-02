import numpy as np


from num_util.const_util import *


def add_new_dim(arr, axis, n):
    '''
    Add a new dimension to the array in a axis, the number of elements in the new dimension is n 
    '''
    np.stack( [arr]*n, axis=axis )
    return arr