# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import os

# @author ZhangMinghui,Southeast University
#
# Module Function: CurvePloter

CurvePloter_DefaultConig={
    'figsize':(7,5),
    'dpi':150,
    'font_family':'Times New Roman',
    'tricks':np.array([0.2,0.5,0.7,0.75,0.8,0.85,0.88,0.92]),
    'tricks_size':10,
    'legend_size':10,
    'label_size':14,
    'lw':1,
    'smooth_number':15,
    'smooth_factor':0.80
}

class CurvePloter(object):
    def __init__(self,Filepath,LabelName,Smoothtype,Number,PltConfig=CurvePloter_DefaultConig,SavePath=None):
        """
        :param Filepath: Lists
        :param LabelName: Lists
        :param LegendName: Lists
        :param PltConfig: CurvePloter_DefaultConig
        """
        self.Filepath = Filepath
        self.LabelName = LabelName
        self.PltConfig = PltConfig
        self.SavePath = SavePath
        self.Number = Number
        self.LegendName=[]
        self.X=[]
        self.Y=[]
        self.smooth_type = Smoothtype
        return None

    def __plot__(self):
        self.__load_data__()
        plt.rc('font', family=self.PltConfig['font_family'])
        fig = plt.figure(figsize=self.PltConfig['figsize'], dpi=self.PltConfig['dpi'])
        plt.grid(ls='--', axis="y")
        plt.xticks(fontsize=self.PltConfig['tricks_size'])
        plt.yticks(fontsize=self.PltConfig['tricks_size'])
        plt.yticks(self.PltConfig['tricks'])
        for i in range(0,len(self.X)):
            plt.plot(self.X[i],self.Y[i],label=self.LegendName[i],lw=self.PltConfig['lw'])
            plt.legend(prop={'size': self.PltConfig['legend_size']})
        plt.xlabel(self.LabelName[0], size=self.PltConfig['label_size'])
        plt.ylabel(self.LabelName[1], size=self.PltConfig['label_size'])
        if self.SavePath is not None:
            self.__save__(self.SavePath)
        else:
            plt.show()
        return None

    def __save__(self,SavePath):
        plt.savefig(SavePath,bbox_inches='tight',pad_inches=0.0)
        return None

    def __smooth__(self,x,smoothtype):
        if (self.smooth_type == 'MA'):
            WSZ = self.PltConfig['smooth_number']
            out0 = np.convolve(x, np.ones(WSZ, dtype=int), 'valid') /WSZ
            r = np.arange(1, WSZ - 1, 2)
            start = np.cumsum(x[:WSZ - 1])[::2] / r
            stop = (np.cumsum(x[:-WSZ:-1])[::2] / r)[::-1]
            return np.concatenate((start, out0, stop))
        if (self.smooth_type == 'EA'):
            factor = self.PltConfig['smooth_factor']
            smoothed_points = []
            for point in x:
                if smoothed_points:
                    previous = smoothed_points[-1]
                    smoothed_points.append(previous * factor + point * (1 - factor))
                else:
                    smoothed_points.append(point)
            return smoothed_points
        return None

    def __load_data__(self):
        FileNameLists = os.listdir(self.Filepath)
        FileLists = [os.path.join(self.Filepath,File) for File in FileNameLists]
        self.X = [np.loadtxt(File)[0:self.Number,0] for File in FileLists]
        self.Y = [np.loadtxt(File)[:,1] for File in FileLists]
        if self.smooth_type is not None:
            self.Y = [self.__smooth__(y,self.smooth_type)[0:self.Number] for y in self.Y]
        self.LegendName = [name[2:-4] for name in FileNameLists]
        return None




if __name__ == "__main__":
    InputConfig = {
        'FilePath':"E:\Studyfile_ZMH\GraduationProject\Important\AAA_Graduation_Important_log\\WholeMetric\\vnet\\hvsmr\\",
        'LabelName':['Epochs','Test Mean Dice'],
        'Smoothtype':'EA',
        'SavePath':None,
        'Number':100
    }

    Ploter = CurvePloter(Filepath=InputConfig['FilePath'],
                         LabelName=InputConfig['LabelName'],
                         Smoothtype=InputConfig['Smoothtype'],
                         SavePath=InputConfig['SavePath'],
                         Number=InputConfig['Number'])
    Ploter.__plot__()