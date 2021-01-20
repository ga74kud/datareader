# -------------------------------------------------------------
# code developed by Michael Hartmann during his Ph.D.
# Data Processing: __init__.py
#
# (C) 2021 Michael Hartmann, Graz, Austria
# Released under GNU GENERAL PUBLIC LICENSE
# email michael.hartmann@v2c2.at
# -------------------------------------------------------------
from datareader.source.preprocessing.dataprocessing import *
from datareader.source.util.signalanalysis import *
from datareader.definitions import *

'''
    Some parameters
'''
def get_params():
    params=dict()
    params['window_x'] = 101
    params['window_y'] = 101
    params['poly_x'] = 2
    params['poly_y'] = 2
    params['PROJECT_ROOT']=read_project_root
    return params


'''
    Get the project root (see definitions.py)
'''
def read_project_root():
    return get_project_root_pkg()

'''
    Get the dataprocessing object
'''
def get_object(params):
    return dataprocessing(params)

'''
    Read the dataset
'''
def read_dataset(params, path):
    obj = dataprocessing(params)
    X=obj.readDataset(path)
    return X

'''
   Example function for dataset
'''
def read_dataset_pkg():
    import os
    import datareader
    params = datareader.get_params()
    path = datareader.__path__[0]
    newpath = os.path.join(path, "data/input/stanford/bookstore/video0/annotations.txt")
    X = datareader.read_dataset(params, newpath)
    print(X)

'''
   Get velocity and acceleration for 2D dataset
'''
def get_velocity_accleration_2D(dataset, params):
    return get_velocity_acceleration_for_dataset_2D(dataset, params)

'''
   Get selection of dataset specified by a column and the range from min_val to max_val
   Example: the dataset should be selected by the time range from [4s, 6s]
'''
def get_dataset_by_range(dataset, column, min_val, max_val):
    return dataset_by_range(dataset, column, min_val, max_val)