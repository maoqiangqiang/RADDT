import torch 
import numpy as np
import pandas as pd
import time
import json
import sklearn.metrics as metrics

import sys
sys.path.append('./src/')


from dataset import loadDataset
from warmStart import CARTClfWarmStart
from treeFunc import readTreePath, getPredY
from RADDT import  multiStartTreeOptbyGRAD_withC



if __name__ == "__main__":
    

    ################## main Code ##################
    # torch.autograd.set_detect_anomaly(True)

    ## Args 
    dataNumStart = int(sys.argv[1])                 # e.g. 1
    dataNumEnd = int(sys.argv[2])                   # e.g. 1
    runsNumStart = int(sys.argv[3])                 # e.g. 1
    runsNumEnd = int(sys.argv[4])                   # e.g. 1

    treeDepth = int(sys.argv[5])                    # e.g. 2 4 8 
    epochNum = int(sys.argv[6])                     # e.g. 1000; larger than 21 epoch 
    deviceArg =  str(sys.argv[7])                   # "cuda" or "cpu"
    device = torch.device(deviceArg)
    startNum = int(sys.argv[8])                     # e.g. 1, 2, 3, 4, 5...
    numScale = int(sys.argv[9])                     # e.g. 1, 2, 3, 4, 5...



    datasetPath = "./data/"
    DatasetsNames = ["banknote-authentication"]

    
    datasetNum = len(DatasetsNames)
    print("Starting: Total {} datasets".format(datasetNum))
    
    
    # read the treePath from the HDF5 file
    indices_flags_dict = readTreePath(treeDepth, device)


    for datasetIdx in range(dataNumStart-1, dataNumEnd):
        print("############# Dataset[{}]: {} #############".format(datasetIdx+1, DatasetsNames[datasetIdx]))
        for run in range(runsNumStart, runsNumEnd+1):
            print("####### Run: {} #######".format(run))
            torch.manual_seed(run)
            np.random.seed(run)

            data_train, data_valid, data_test = loadDataset(DatasetsNames[datasetIdx], run, datasetPath)
            p = data_train.shape[1] - 1
            X_train = torch.from_numpy(data_train[:, 0:p] * 1.0).float()
            Y_train = torch.from_numpy(data_train[:, p]).long()
            X_valid = torch.from_numpy(data_valid[:, 0:p] * 1.0).float()
            Y_valid = torch.from_numpy(data_valid[:, p]).long()
            X_test = torch.from_numpy(data_test[:, 0:p] * 1.0).float()
            Y_test = torch.from_numpy(data_test[:, p]).long()
            X = torch.cat((X_train, X_valid), 0)
            Y = torch.cat((Y_train, Y_valid), 0)
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            # X_train = X_train.to(device, non_blocking=True)
            # Y_train = Y_train.to(device, non_blocking=True)
            # X_valid = X_valid.to(device, non_blocking=True)
            # Y_valid = Y_valid.to(device, non_blocking=True)
            X_test = X_test.to(device, non_blocking=True)
            Y_test = Y_test.to(device, non_blocking=True)
            # X_all = torch.cat((X, X_test), 0)
            Y_all = torch.cat((Y, Y_test), 0)

            if run == runsNumStart:
                print("dataset:{};    n_train:{};    n_valid:{};    n_test:{};    p:{}\n".format(DatasetsNames[datasetIdx], X_train.shape[0], X_valid.shape[0], X_test.shape[0], X_train.shape[1]))


            startTime = time.perf_counter()
            
            nClass = torch.unique(Y_all).shape[0]
            print(f"nClass: {nClass}")
            # cart warm start
            aInit, bInit, cInit = CARTClfWarmStart(X, Y, treeDepth, nClass, device)
            cartWarmStart_dict = {"a": aInit, "b": bInit, "c": cInit}
            warmStart = [cartWarmStart_dict]
            acc_DDTCur, treeDDTCur = multiStartTreeOptbyGRAD_withC(X, Y, treeDepth, nClass, indices_flags_dict, epochNum, device, warmStart, startNum, numScale)

            elapsedTime = time.perf_counter() - startTime

            # get the predY
            Y_Pred = getPredY(X, Y, treeDepth, treeDDTCur)
            Y_Prednp = Y_Pred.cpu().numpy()
            Y_test_Pred = getPredY(X_test, Y_test, treeDepth, treeDDTCur)
            Y_test_Prednp = Y_test_Pred.cpu().numpy()
            Y_np = Y.cpu().numpy()
            Y_test_np = Y_test.cpu().numpy()
    
            acc_train = metrics.accuracy_score(Y_np, Y_Prednp)
            acc_test = metrics.accuracy_score(Y_test_np, Y_test_Prednp)
            f1_train = metrics.f1_score(Y_np, Y_Prednp, average='macro')
            f1_test = metrics.f1_score(Y_test_np, Y_test_Prednp, average='macro')

            ## final results
            print("\nFinal Results...")
            print("acc_train: {};   acc_test: {};   f1_train: {};   f1_test: {}".format(acc_train, acc_test, f1_train, f1_test))
            print("elapsedTime: {}".format(elapsedTime))


