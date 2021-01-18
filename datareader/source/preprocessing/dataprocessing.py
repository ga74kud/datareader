from source.util.signalanalysis import *
from source.util.helpfunctions import *

from pathlib import Path


class dataprocessing(object):
    def __init__(self, params, **kwargs):
        self.ROOT_DIR = str(params['PROJECT_ROOT'])
        self.file_path = self.readInputDataTxt()
        self.dataset = self.readDataset()
        #param2DataFrame = gettingAlgorithmicParams(self.ROOT_DIR)
        # self.dataDict=self.computeDataDict(params, param2DataFrame)
        self.dataset = self.datasetWithProcessing(params)
        self.extremePositions = self.getExtrema()

    def getExtrema(self):
        xmin, xmax, ymin, ymax = computeExtrema(self.dataset)
        extremePositions = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}

        return extremePositions

    def get_project_root(self):
        """Returns project root folder."""
        return Path(__file__).parent.parent.parent

    def selectFilePath(self):
        returnPath = str(self.ROOT_DIR) + "/data/input/stanford/bookstore/video0/"
        return returnPath

    def readInputDataTxt(self):
        data_path = self.selectFilePath()
        file_path = data_path + 'annotations.txt'
        return file_path

    def readDataset(self):
        print(self.file_path)
        df = pd.read_csv(self.file_path, sep=" ", header=None,
                         names=['id', 'xmin', 'ymin', 'xmax', 'ymax', 't', 'a2', 'a3', 'a4', 'label'])
        # data = pd.read_csv(self.file_path, header=None)
        return df

    def getOutputDataPath(self, name):
        filepath = str(self.ROOT_DIR) + "/data/output/" + name
        return filepath

    def datasetWithProcessing(self, params):
        knowledgeDataset = getSignalFromMultipleSources(self.dataset, params)
        return knowledgeDataset

# print(a.dataDict['xcent'])