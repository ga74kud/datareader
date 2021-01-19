import os

from datareader.source.preprocessing.dataprocessing import *
from datareader.definitions import *
def get_params():
    params=dict()
    params['window_x'] = 101
    params['window_y'] = 101
    params['poly_x'] = 2
    params['poly_y'] = 2
    params['PROJECT_ROOT']=read_project_root
    return params

def read_project_root():
    return get_project_root_pkg()

def get_object(params):
    return dataprocessing(params)

def read_dataset(params, path):
    obj = dataprocessing(params)
    X=obj.readDataset(path)
    return X

def read_params(params):
    print(params)

def read_dataset_pkg():
    import os
    import datareader
    params = datareader.get_params()
    path = datareader.__path__[0]
    newpath = os.path.join(path, "data/input/stanford/bookstore/video0/annotations.txt")
    X = datareader.read_dataset(params, newpath)
    print(X)

