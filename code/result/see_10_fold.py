import pandas as pd
import glob
import sys
import os
import time
import numpy as np
allFileFaList = glob.glob(r"F:\Download\lengent\venv\MAHyNet-main\demo\result_local\*")

def see(allFileFaList):
    cour = []
    dataset=[]
    mode=[]
    countauc=0
    for FilePath in allFileFaList:
        filenames = glob.glob(FilePath + "/*.npy")
        data = FilePath.split("/")[-1]
        for allFileFa in filenames:
            countauc+=1
            mode.extend([allFileFa.split("_")[-1].split(".")[0]])
            AUC = np.load( allFileFa, encoding='bytes', allow_pickle=True)
            cour.append(AUC)
            dataset.append(data)
            AUCs = np.mean(cour)
            print(AUCs,countauc)
            dic = {
                'dataset':dataset,
                'AUC': cour,
                'mode':mode
            }
            # list to DataFrame
            data_all = pd.DataFrame(dic)
            data_all['dataset'] = data_all['dataset']
            data_all['AUC'] = data_all['AUC'].astype(float)
            data_all['mode'] = data_all['mode']
            data_all.to_csv('result_10.csv', index=False)

see(allFileFaList)
