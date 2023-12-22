# basic routine for reading data
import numpy as np

def read_csv(filename, delimiter=','):
    '''
    Routine to read csv data
    '''
    import csv
    dat = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for row in csv_reader:
            dat.append(row)
    return dat
