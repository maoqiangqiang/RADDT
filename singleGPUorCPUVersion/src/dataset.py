import pandas as pd 

def loadDataset(datasetName, run, dirPath):
    dataPath = dirPath + datasetName + "_" + str(run) + "_"
    dataTrain = pd.read_csv(dataPath+"train.csv", header=0)
    dataTrain = dataTrain.to_numpy()
    dataValid = pd.read_csv(dataPath+"valid.csv", header=0)
    dataValid = dataValid.to_numpy()
    dataTest = pd.read_csv(dataPath+"test.csv", header=0)
    dataTest = dataTest.to_numpy()
    return dataTrain, dataValid, dataTest