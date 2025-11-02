import sys, os
import traceback
import numpy as np

def print_full_array(arr):
    '''
    Print the entire array
    '''
    np.set_printoptions(threshold=np.inf)
    print(arr)
    np.set_printoptions()


def print_Exception(e, *args):
    print(args)
    print(f'error message: {e}')
    
    # Retrieve the full traceback as a list of strings
    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
    
    # Print the full traceback
    for line in tb_lines:
        print(line, end='')

print_exception = print_Exception
''' 
def print_Exception(e, *args):
    print(args)
    print(f'error message: {e}' )
    print(f"File: {e.__traceback__.tb_frame.f_globals['__file__']}")
    print(f"Line #: {e.__traceback__.tb_lineno}")
''' 