

import math
import sys

import numpy as np
import pandas as pd
#import pylab as py
from scipy.signal import savgol_filter
import _pickle as cPickle
def sg_filter(x, m, k=0):
    """
    code found https://dsp.stackexchange.com/questions/9498/have-position-want-to-calculate-velocity-and-acceleration
    12. October 2019

    Example Usage:
    python sg.py position.dat 7 2

    x = Vector of sample times
    m = Order of the smoothing polynomial
    k = Which derivative
    """
    mid = len(x) / 2
    a = x - x[mid]
    expa = lambda x: map(lambda i: i**x, a)
    A = np.r_[map(expa, range(0,m+1))].transpose()
    Ai = np.linalg.pinv(A)

    return Ai[k]

def smooth(x, y, size=5, order=2, deriv=0):
    """
        code found https://dsp.stackexchange.com/questions/9498/have-position-want-to-calculate-velocity-and-acceleration
        12. October 2019

        Example Usage:
        python sg.py position.dat 7 2
    """
    if deriv > order:
        raise AssertionError

    n = len(x)
    m = size

    result = np.zeros(n)

    for i in range(m, n-m):
        start, end = i - m, i + m + 1
        f = sg_filter(x[start:end], order, deriv)
        result[i] = np.dot(f, y[start:end])

    if deriv > 1:
        result *= math.factorial(deriv)

    return result

def plot(t, plots):
    n = len(plots)

    for i in range(0,n):
        label, data = plots[i]

     #   plt = py.subplot(n, 1, i+1)
     #   plt.tick_params(labelsize=8)
     #   py.grid()
     #   py.xlim([t[0], t[-1]])
     #   py.ylabel(label)

     #   py.plot(t, data, 'k-')

    #py.xlabel("Time")

def create_figure(size, order):
    #fig = py.figure(figsize=(8,6))
    nth = 'th'
    if order < 4:
        nth = ['st','nd','rd','th'][order-1]

    title = "%s point smoothing" % size
    title += ", %d%s degree polynomial" % (order, nth)

    #fig.text(.5, .92, title,
    #         horizontalalignment='center')

def findValueDict(dict, sourcecolumn, targetcolumn, keyword):
    arrayLineIdx=dict.loc[dict[sourcecolumn] == keyword].index.values
    columnIdx=dict[targetcolumn]
    value=int(columnIdx[arrayLineIdx])
    return value

def getSignalsByDataFrame(data, param, param2):
    x = np.array(data[:, 0])
    y = np.array(data[:, 1])
    t = np.array(data[:, 2])
    windowX=findValueDict(param2, 'name', 'value', 'windowX')
    windowY=findValueDict(param2, 'name', 'value', 'windowY')
    polyOrderX=findValueDict(param2, 'name', 'value', 'polyX')
    polyOrderY=findValueDict(param2, 'name', 'value', 'polyY')
    plotsX = [
        ["x-Position",     savgol_filter(x, windowX, polyOrderX)],
        ["x-Velocity",     savgol_filter(x, windowX, polyOrderX, 1)],
        ["x-Acceleration", savgol_filter(x, windowX, polyOrderX, 2)]
    ]
    plotsY = [
        ["y-Position", savgol_filter(y, windowY, polyOrderY)],
        ["y-Velocity", savgol_filter(y, windowY, polyOrderY, 1)],
        ["y-Acceleration", savgol_filter(y, windowY, polyOrderY, 2)]
    ]
    return plotsX, plotsY, t

def plot_results(plotsX, plotsY, t):
    create_figure(11, 2)
    plot(t, plotsX)
    create_figure(11, 2)
    plot(t, plotsY)


def computeVelocityAcceleration(x, y, params):
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
def getSignalFromMultipleSources(dataset, params):
    allID=np.unique(dataset['id'])
    df = pd.DataFrame(columns=['t', 'id', 'px', 'py', 'vx', 'vy', 'ax', 'ay'])
    frames=None
    for selectedID in allID:
        newFrame = pd.DataFrame(columns=['t', 'id', 'px', 'py', 'vx', 'vy', 'ax', 'ay'])
        selDat = dataset.loc[dataset['id'] == selectedID]
        newFrame['t'] = np.transpose(selDat['t'])
        xCent = selDat['xmax'] - selDat['xmin']
        yCent = selDat['ymax'] - selDat['ymin']
        newFrame['px'], newFrame['py'], newFrame['vx'], newFrame['vy'], newFrame['ax'], newFrame['ay']=computeVelocityAcceleration(xCent, yCent, params)

        newFrame['id']=selectedID
        df=pd.concat((df, newFrame))


    return df
