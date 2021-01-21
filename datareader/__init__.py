# -------------------------------------------------------------
# code developed by Michael Hartmann during his Ph.D.
# Data Processing: __init__.py
#
# (C) 2021 Michael Hartmann, Graz, Austria
# Released under GNU GENERAL PUBLIC LICENSE
# email michael.hartmann@v2c2.at
# -------------------------------------------------------------
import logging
import cv2
from datareader.source.preprocessing.dataprocessing import *
from datareader.source.util.signalanalysis import *
from datareader.definitions import *


'''
    Some parameters
'''
def get_params():
    params=dict()
    params['window_x'] = 11
    params['window_y'] = 11
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
    X=obj.read_dataset(path)
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
def get_velocity_for_dataset_2D(dataset, params):
    return get_velocity_signal_for_dataset_2D(dataset, params)

'''
   Get selection of dataset specified by a column and the range from min_val to max_val
   Example: the dataset should be selected by the time range from [4s, 6s]
'''
def get_dataset_by_range(dataset, column, min_val, max_val):
    return dataset_by_range(dataset, column, min_val, max_val)

'''
   Get zonotypes from dataset by a row idx
'''
def get_zonotypes_by_row(dataset, row_idx):
    return get_zonotypes_2D_by_row(dataset, row_idx)

'''
   Get dataset by timestamp
'''
def get_dataset_by_timestamp(dataset, timestamp):
    return get_dataset_by_stamp_by_column(dataset, 't', timestamp)

'''
   Get successor by timestamp for an agent with id
'''
def get_successor_by_timestamp(dataset,timestamp, id):
    #selection by timestamp
    x,a = get_dataset_by_timestamp(dataset, timestamp)
    x,b = get_dataset_by_timestamp(dataset, timestamp+1)
    bool_a, val_a = get_dataset_by_stamp_by_column(a, 'id', id)
    bool_b, val_b = get_dataset_by_stamp_by_column(b, 'id', id)
    if(bool_a==False):
        logging.warn("id not in array at timestamp: " + str(timestamp))
        return 0
    elif(bool_b == False):
        logging.warn("id not in array at timestamp: " + str(timestamp+1))
        return 0
    else:
        return (val_a, val_b)

'''
   Read background image
'''
def read_background_picture(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

'''
   Plot the background image
'''
def plot_background_picture(name, path):
    img = read_background_picture(path)
    #img=plot_rectangle(img, (5,5), (202,202), (255, 0, 0), 2)
    show(name, img)

'''
   Show image
'''
def show(name, img):
    cv2.imshow(name, img)

'''
   Show image and hold the image
'''
def show_and_hold(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)

'''
    plot a rectangle on the image
'''
def plot_rectangle(image, start_point, end_point, color, thickness):
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image

'''
    plot all rectangles on the image for a specific timestamp
'''
def plot_rectangle_for_timestamp(img, dataset, timestamp):
    no, XY = get_dataset_by_timestamp(dataset, timestamp)
    for index, row in XY.iterrows():
        img=plot_rectangle(img, (row['xmin'],row['ymin']), (row['xmax'],row['ymax']), (255, 0, 0), 2)
    return img