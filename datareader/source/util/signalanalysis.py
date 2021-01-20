# -------------------------------------------------------------
# code developed by Michael Hartmann during his Ph.D.
# Data Processing: Signal Analysis
#
# (C) 2021 Michael Hartmann, Graz, Austria
# Released under GNU GENERAL PUBLIC LICENSE
# email michael.hartmann@v2c2.at
# -------------------------------------------------------------

import math
import sys

import numpy as np
import pandas as pd
#import pylab as py
from scipy.signal import savgol_filter
import _pickle as cPickle


# """
#     code copied https://dsp.stackexchange.com/questions/9498/have-position-want-to-calculate-velocity-and-acceleration
#     12. October 2019
#     "Given Position measurements, how to estimate velocity, acceleration"
#
#     Example Usage:
#     python sg.py position.dat 7 2
#
#     x = Vector of sample times
#     m = Order of the smoothing polynomial
#     k = Which derivative
# """
# def sg_filter(x, m, k=0):
#
#     mid = len(x) / 2
#     a = x - x[mid]
#     expa = lambda x: map(lambda i: i**x, a)
#     A = np.r_[map(expa, range(0,m+1))].transpose()
#     Ai = np.linalg.pinv(A)
#
#     return Ai[k]
# """
#         code copied https://dsp.stackexchange.com/questions/9498/have-position-want-to-calculate-velocity-and-acceleration
#         12. October 2019
#
#         Example Usage:
#         python sg.py position.dat 7 2
# """
# def smooth(x, y, size=5, order=2, deriv=0):
#
#     if deriv > order:
#         raise AssertionError
#
#     n = len(x)
#     m = size
#
#     result = np.zeros(n)
#
#     for i in range(m, n-m):
#         start, end = i - m, i + m + 1
#         f = sg_filter(x[start:end], order, deriv)
#         result[i] = np.dot(f, y[start:end])
#
#     if deriv > 1:
#         result *= math.factorial(deriv)
#
#     return result



# """
#
# """
#
# def findValueDict(dict, sourcecolumn, targetcolumn, keyword):
#     arrayLineIdx=dict.loc[dict[sourcecolumn] == keyword].index.values
#     columnIdx=dict[targetcolumn]
#     value=int(columnIdx[arrayLineIdx])
#     return value

# def getSignalsByDataFrame(data, param, param2):
#     x = np.array(data[:, 0])
#     y = np.array(data[:, 1])
#     t = np.array(data[:, 2])
#     windowX=findValueDict(param2, 'name', 'value', 'windowX')
#     windowY=findValueDict(param2, 'name', 'value', 'windowY')
#     polyOrderX=findValueDict(param2, 'name', 'value', 'polyX')
#     polyOrderY=findValueDict(param2, 'name', 'value', 'polyY')
#     plotsX = [
#         ["x-Position",     savgol_filter(x, windowX, polyOrderX)],
#         ["x-Velocity",     savgol_filter(x, windowX, polyOrderX, 1)],
#         ["x-Acceleration", savgol_filter(x, windowX, polyOrderX, 2)]
#     ]
#     plotsY = [
#         ["y-Position", savgol_filter(y, windowY, polyOrderY)],
#         ["y-Velocity", savgol_filter(y, windowY, polyOrderY, 1)],
#         ["y-Acceleration", savgol_filter(y, windowY, polyOrderY, 2)]
#     ]
#     return plotsX, plotsY, t


"""
    Compute velocity and acceleration with savgol_filter-function
    ("https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html")
"""
def compute_velocity_acceleration_with_filter(x, y, params):
    windowX = params['window_x']
    windowY = params['window_y']
    polyOrderX = params['poly_x']
    polyOrderY = params['poly_y']
    px=savgol_filter(x, windowX, polyOrderX)
    vx=savgol_filter(x, windowX, polyOrderX, 1)
    ax=savgol_filter(x, windowX, polyOrderX, 2)
    py=savgol_filter(y, windowY, polyOrderY)
    vy=savgol_filter(y, windowY, polyOrderY, 1)
    ay=savgol_filter(y, windowY, polyOrderY, 2)
    return px, py, vx, vy, ax, ay


"""
    Compute for the dataset (pandas-dataframe) velocity and acceleration (function compute_velocity_acceleration_with_filter)
"""
def get_velocity_acceleration_for_dataset_2D(dataset, params):
    allID=np.unique(dataset['id'])
    df = pd.DataFrame(columns=['t', 'id', 'px', 'py', 'vx', 'vy', 'ax', 'ay'])
    for selectedID in allID:
        newFrame = pd.DataFrame(columns=['t', 'id', 'px', 'py', 'vx', 'vy', 'ax', 'ay'])
        selDat = dataset.loc[dataset['id'] == selectedID]
        newFrame['t'] = np.transpose(selDat['t'])
        xCent = selDat['xmax'] - selDat['xmin']
        yCent = selDat['ymax'] - selDat['ymin']
        newFrame['px'], newFrame['py'], newFrame['vx'], newFrame['vy'], newFrame['ax'], newFrame['ay']=compute_velocity_acceleration_with_filter(xCent, yCent, params)

        newFrame['id']=selectedID
        df=pd.concat((df, newFrame))
    return df
