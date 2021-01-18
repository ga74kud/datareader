import numpy as np
from pathlib import Path
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import pandas as pd


def loadMatFileTooPandasDataframe(filepath):
    mat = loadmat(filepath)  # load mat-file
    mdata = mat['myDat']  # variable in mat file
    df = pd.DataFrame(mdata)
    return df

def computeMatrixForHeightmap(dataset):
    return 0
def computeExtrema(dataset):
    xmin=np.min(dataset['px'])
    xmax=np.max(dataset['px'])
    ymin = np.min(dataset['py'])
    ymax = np.max(dataset['py'])
    return xmin, xmax, ymin, ymax

def computeRegularTopologicalSpace(extremePositions):
    xmin = extremePositions['xmin']
    xmax = extremePositions['xmax']
    ymin = extremePositions['ymin']
    ymax = extremePositions['ymax']
    return 0

def get_project_root():
        """Returns project root folder."""
        return Path(__file__).parent.parent.parent
def selectFilePath(dataPath):
    returnPath= str(get_project_root()) + dataPath
    return  returnPath
def readInputDataTxt(dataPath, dataFile):
    data_path = selectFilePath(dataPath)
    file_path=data_path+dataFile
    return file_path

def gettingAlgorithmicParams(projectRoot):
    paramPath=str(projectRoot) + "/params/params.txt"
    param2DataFrame=pd.read_csv(paramPath, header=0, sep='|', lineterminator='\n')
    param2={'name': param2DataFrame['name'], 'value': param2DataFrame['value'], 'description': param2DataFrame['description']}
    return param2DataFrame

def getParamFromPandas(df, row):
    line=df['name']==row
    selLineIdx=df.index[line == True].tolist()
    selLineIdx=selLineIdx[0]
    test=df.iloc[selLineIdx]
    returnValue=test['value']
    return returnValue