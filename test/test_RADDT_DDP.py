import torch 
import numpy as np
import pandas as pd
import time
import json
import argparse

import sys
sys.path.append('./src/')

from dataset import loadDataset 
from treeFunc_DDP import readTreePath, objv_cost

from warmStart import CARTRegWarmStart
from RADDT_DDP import  multiStartTreeOptbyGRAD_withC

import distDataParallel 


if __name__ == "__main__":


    ################## main Code ##################
    # torch.autograd.set_detect_anomaly(True)

    totalStart = time.perf_counter()

    # parse arguments 

    parser = argparse.ArgumentParser(description='RADDT DDP')
    parser.add_argument('--dist-backend', default='gloo', type=str, help='')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument("--dataNumStart", default=1, type=int)
    parser.add_argument("--dataNumEnd", default=1, type=int)
    parser.add_argument("--runsNumStart", default=1, type=int)
    parser.add_argument("--runsNumEnd", default=1, type=int)
    parser.add_argument("--treeDepth", default=8, type=int)
    parser.add_argument("--epochNum", default=3000, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--startNum", default=10, type=int)
    parser.add_argument("--numScale", default=5, type=int)
    parser.add_argument("--csvOutputFlag", default=1, type=bool)
    
    args = parser.parse_args()

    dataNumStart = args.dataNumStart
    dataNumEnd = args.dataNumEnd
    runsNumStart = args.runsNumStart
    runsNumEnd = args.runsNumEnd
    treeDepth = args.treeDepth
    epochNum = args.epochNum    
    deviceArg = args.device

    startNum = args.startNum
    numScale = args.numScale
    csvOutputFlag = args.csvOutputFlag




    ##  data
    datasetPath = "../data/"
    DatasetsNames = ["airfoil-self-noise", "space-ga", "abalone", "gas-turbine-co-emission-2015", "gas-turbine-nox-emission-2015",  "puma8NH",  "cpu-act", "cpu-small", "kin8nm", "delta-elevators", "combined-cycle-power-plant", "electrical-grid-stability", "condition-based-maintenance_compressor", "condition-based-maintenance_turbine", "ailerons", "elevators","friedman-artificial", "BNG_Ailerons", "BNG_cpu_act","BNG_cpu_small", "BNG_puma32H", "BNG_wisconsin", "ACSPublicCoverage2018","BNG_elevators"]




    rank, localRank =  distDataParallel.setup(args.dist_backend, args.init_method, args.world_size)
    print("The rank is {} and the current device is {}".format(rank, localRank))
            
    # device = torch.device(deviceArg)
    device = torch.device(f"cuda:{localRank}")




    # read the treePath from the HDF5 file
    indices_flags_dict = readTreePath(treeDepth, device)

    datasetNum = len(DatasetsNames)
    print("Starting: Total {} datasets".format(datasetNum))






    for datasetIdx in range(dataNumStart-1, dataNumEnd):
        print(f"Rank {rank} ############# Dataset[{datasetIdx + 1}]: {DatasetsNames[datasetIdx]} #############")
        # print("############# Dataset[{}]: {} #############".format(datasetIdx+1, DatasetsNames[datasetIdx]))
        for run in range(runsNumStart, runsNumEnd+1):
            print(f"Rank {rank} ####### Run: {run} #######")
            # print("####### Run: {} #######".format(run))
            torch.manual_seed(run)
            np.random.seed(run)


            dataTrain, dataValid, dataTest = loadDataset(DatasetsNames[datasetIdx], run, datasetPath)
            p = dataTrain.shape[1] - 1
            X_train = torch.from_numpy(dataTrain[:, 0:p] * 1.0).float()
            Y_train = torch.from_numpy(dataTrain[:, p] * 1.0).float()
            X_valid = torch.from_numpy(dataValid[:, 0:p] * 1.0).float()
            Y_valid = torch.from_numpy(dataValid[:, p] * 1.0).float()
            X_test = torch.from_numpy(dataTest[:, 0:p] * 1.0).float()
            Y_test = torch.from_numpy(dataTest[:, p] * 1.0).float()
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


            ### split data by rank 
            totalSamples = X.shape[0]
            samplesPerRank = totalSamples // args.world_size
            startIdx = rank * samplesPerRank
            endIdx = startIdx + samplesPerRank if rank != args.world_size - 1 else totalSamples

            X_rank = X[startIdx:endIdx].to(device, non_blocking=True)
            Y_rank = Y[startIdx:endIdx].to(device, non_blocking=True)



            if run == runsNumStart and rank == 0:
                print("dataset:{};    n_train:{};    n_valid:{};    n_test:{};    p:{}\n".format(DatasetsNames[datasetIdx], X_train.shape[0], X_valid.shape[0], X_test.shape[0], X_train.shape[1]))
                print("samplesPerRank: {}".format(samplesPerRank))



            startTimeRank = time.perf_counter()
            # cart warm start
            aInit, bInit, cInit = CARTRegWarmStart(X, Y, treeDepth, device)
            cartWarmStart_dict = {"a": aInit, "b": bInit, "c": cInit}
            warmStart = [cartWarmStart_dict]
            if rank == 0:
                objv_GETCur, treeGETCur = multiStartTreeOptbyGRAD_withC(X_rank, Y_rank, X, Y, treeDepth, indices_flags_dict, epochNum, device, warmStart, startNum, numScale, rank, localRank)
            else:
                _ = multiStartTreeOptbyGRAD_withC(X_rank, Y_rank, X, Y, treeDepth, indices_flags_dict, epochNum, device, warmStart, startNum, numScale, rank, localRank)

            elapsedTimeRank = time.perf_counter() - startTimeRank
            print(f"Rank {rank} elapsed time: {elapsedTimeRank}")

            # Synchronize before saving (optional)
            torch.distributed.barrier() 

            
            elapsedTime = time.perf_counter() - totalStart  

            if rank == 0:
                objvMseTrainGET, r2TrainGET = objv_cost(X, Y, treeDepth, treeGETCur)
                objvMseTestGET, r2TestGET = objv_cost(X_test, Y_test, treeDepth, treeGETCur)

                print("\nFinal Results...")
                print("objvMseTrainGET: {};   r2TrainGET: {}".format(objvMseTrainGET, r2TrainGET))
                print("objvMseTestGET: {};   r2TestGET: {}".format(objvMseTestGET, r2TestGET))

                print("elapsedTimeCV: {}".format(elapsedTime))


                

        torch.distributed.barrier()
    distDataParallel.cleanup()


