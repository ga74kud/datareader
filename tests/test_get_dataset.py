# -------------------------------------------------------------
# code developed by Michael Hartmann during his Ph.D.
# Data Processing: Test script for ensuring functionality of python package datareader
#
# (C) 2021 Michael Hartmann, Graz, Austria
# Released under GNU GENERAL PUBLIC LICENSE
# email michael.hartmann@v2c2.at
# -------------------------------------------------------------

import os
import logging
import datareader
import numpy as np

'''
   Get the parameters
'''
def get_params():
    params = datareader.get_params()
    return params

'''
   Read one stanford dataset
'''
def read_dataset():
    params = get_params()
    path = datareader.__path__[0]
    newpath = os.path.join(path, "data/input/stanford/bookstore/video0/annotations.txt")
    X = datareader.read_dataset(params, newpath)
    logging.info(X)
    return X

'''
   Compute the velocity and acceleration
'''
def compute_velocity_acceleration(X):
    params = get_params()
    Y=datareader.get_velocity_acceleration_for_dataset_2D(X, params)
    logging.info(Y)
    return Y

'''
   Test the functionality of datareader
'''
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    X=read_dataset()
    Y=compute_velocity_acceleration(X)
