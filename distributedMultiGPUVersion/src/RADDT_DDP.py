import torch 
import copy 
import math
import sys 
sys.path.append('./src/')

from warmStart import CARTRegWarmStart
from treeFunc_DDP import objv_cost, update_c, abjustA_b_Compr
from modifiedScheduler import ChainedScheduler
import numpy as np

import os 
##  enforce deterministic behavior in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"




class branchNodeNet(torch.nn.Module):
    def __init__(self, treeDepth: int, p: int, scale: float, a_init: torch.Tensor, b_init: torch.Tensor, c_init: torch.Tensor) -> None:
        super().__init__()
        self.depth = treeDepth
        self.featNum = p
        # self.treesize = 2 ** (self.depth + 1) - 1
        self.branchNodeNum = 2 ** (self.depth) - 1
        self.scale = scale

        self.linear1 = torch.nn.Linear(self.featNum, self.branchNodeNum)
        if a_init is not None:
            self.linear1.weight = torch.nn.Parameter(a_init.clone().detach().requires_grad_(True))
        if b_init is not None:
            self.linear1.bias = torch.nn.Parameter(b_init.clone().detach().requires_grad_(True))
        # Define a ReLU layer
        self.relu = torch.nn.ReLU()

        # leaf labels as trainable parameters 
        if c_init is not None:
            self.c_leafLable = torch.nn.Parameter(c_init.clone().detach().requires_grad_(True))
        else:
            self.c_leafLable = None
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.linear1(x)
        reluBMinusAx = self.relu(-x)
        reluAxMinusB = self.relu(x)
        return reluBMinusAx, reluAxMinusB, self.c_leafLable
    


## the version of matrix operation and broadcasting
@torch.jit.script
def objectiveFuncwithC(scale: float, reluBMinusAx: torch.Tensor, reluAxMinusB: torch.Tensor, Y_train: torch.Tensor, c_leafLable: torch.Tensor,  treeDepth: int, indices_tensor: torch.Tensor, flags_tensor: torch.Tensor, batch_size: int) -> torch.Tensor: 

    N_num = Y_train.shape[0]
    total_objv = torch.tensor(0.0, device=reluBMinusAx.device)
    
    device = reluBMinusAx.device  # Use the same device as reluBMinusAx (part of indicators_stack)
    indices_tensor = indices_tensor.to(device)
    flags_tensor = flags_tensor.to(device)


    for i in range(0, N_num, batch_size):
        batch_End = min(i+batch_size, N_num)
        batch_Y_train = Y_train[i:batch_End]
        batch_Y_train_diff = (batch_Y_train.view(-1, 1) - c_leafLable.view(1, -1)).pow_(2)
        # Stack indicators and their complement
        indicators_stack = torch.stack((reluBMinusAx[i:batch_End,:], reluAxMinusB[i:batch_End,:]))
        # Use broadcasting and advanced indexing to select appropriate indicators
        selected_indicators = indicators_stack[flags_tensor, :, indices_tensor]
        indicator_pairs = selected_indicators.sum(dim=1).transpose(0, 1)

        # add 1 softmax to the Relu-based indicator_pairs
        scaleSoftmax = scale
        indicator_pairs = indicator_pairs * scaleSoftmax
        ReluPitSoftmax = torch.nn.functional.softmin(indicator_pairs, dim=1)

        if treeDepth > 4:
            ReluPitSoftmax = torch.nn.functional.threshold(ReluPitSoftmax, 1.0/(2**treeDepth), 0.0)

        batch_objv = (ReluPitSoftmax * batch_Y_train_diff).sum()
        total_objv += batch_objv
    
    total_objv = total_objv / N_num
    return total_objv




def initialize_parameters(choice, net, warmStarts, X_train, Y_train, device, MultiAbFlag=False):
    
    a, b, c = None, None, None
    if choice < len(warmStarts) and warmStarts[choice] is not None:
        ws = warmStarts[choice]
        a, b, c = copy.deepcopy(ws["a"]), copy.deepcopy(ws["b"]), copy.deepcopy(ws["c"])
        aTensor = torch.as_tensor(a, dtype=torch.float32, device=device)
        bTensor = torch.as_tensor(b, dtype=torch.float32, device=device) * (-1.0)
        c = torch.as_tensor(c, dtype=torch.float32, device=device)
        aMatrixAdjustCompr, negBVectorAdjustCompr = abjustA_b_Compr(X_train, aTensor, bTensor, MultiAbFlag)
        
    else:
        a = torch.rand(net.linear1.weight.shape,dtype=torch.float32, device=device) * (2.0)+(-1.0)
        b = torch.rand(net.linear1.bias.shape, dtype=torch.float32, device=device) * (2.0)+(-1.0)
        aMatrixAdjustCompr, negBVectorAdjustCompr = abjustA_b_Compr(X_train, a, b*(-1.0), MultiAbFlag)
        c = update_c(X_train, Y_train, net.depth, {"a": aMatrixAdjustCompr, "b": negBVectorAdjustCompr})["c"]

    return aMatrixAdjustCompr, negBVectorAdjustCompr, c




class callbackFuncs:
    def CalMSEonEpochEnd(self, X_train, Y_train, treeDepth, netLinear1, net_c_leafLable):
        with torch.no_grad():    
            a_grad = netLinear1.weight.clone().detach() 
            b_grad = netLinear1.bias.clone().detach() * (-1.0)
            c_grad = net_c_leafLable.clone().detach() 
            treeEpoch = {"a": a_grad, "b": b_grad, "c": c_grad}
            
        treeEpoch = update_c(X_train, Y_train, treeDepth, treeEpoch)
        objvMSE_Epoch, r2_Epoch = objv_cost(X_train, Y_train, treeDepth, treeEpoch)
        return objvMSE_Epoch, r2_Epoch, treeEpoch
        



def treeOptbyGRADwithC(treeDepth, indices_flags_dict, epochNum, X_train, Y_train, X, Y, device, warm_starts, scaleFactor, lrscheList,  idxStart, callback, rank, localRank, adjust_ab=False):

    ## hyperparameters
    learningRate, T0, warmupsteps, gamma = lrscheList[0], lrscheList[1], lrscheList[2], lrscheList[3]


    ##  net
    p = X_train.shape[1]
    scale = torch.tensor([scaleFactor], device=device)
    net_placeholder = branchNodeNet(treeDepth, p, scale, None, None, None).to(device, non_blocking=True)

    ### initialize weight and bias of net
    a, b, c = initialize_parameters(idxStart, net_placeholder, warm_starts, X_train, Y_train, device, adjust_ab)
    net = branchNodeNet(treeDepth, p, scale, a, b, c).to(device, non_blocking=True)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[localRank])

    optimizer = torch.optim.AdamW(net.module.parameters(), lr=learningRate)



    # ############################################
    # ########### if to disable the scheduler, must comment the following code; otherwise, the lr will be affected ############ 
    # restartNum = 1
    # # Sn = 2**restartNum-1
    # Sn = restartNum
    # warmupsteps = 10
    # T0= math.ceil((epochNum - warmupsteps)/Sn)
    # # print("T0 is {} and warmupsteps is {}".format(T0, warmupsteps))
    # epochNumNew = T0*Sn+warmupsteps
    # # print("epochNumNew is {}".format(epochNumNew))
    # gamma = (1/restartNum)**(1/(restartNum-1))   if restartNum != 1 else 1
    # # print("gamma is {}".format(gamma))

    scheduler = ChainedScheduler(optimizer, T_0=T0, T_mul=1, eta_min=1e-6, max_lr=learningRate, warmup_steps=warmupsteps, gamma=gamma)
    # load the indices_tensor and flags_tensor from the indices_flags_dict

    indices_tensor = indices_flags_dict["D"+str(treeDepth)]["indices_tensor"]
    flags_tensor = indices_flags_dict["D"+str(treeDepth)]["flags_tensor"]

    ## only compare and used by rank 0
    if rank == 0:
        objvMSE_EpochBest = float('inf')
        r2_EpochBest = float('-inf')
        tree_EpochBest = None

    for epoch in range(epochNum):

        optimizer.zero_grad(set_to_none=True)   

        # Forward pass with a batch
        reluBMinuxAx, reluAxMinusB, c_leafLable  = net(X_train)
        objv = objectiveFuncwithC(scaleFactor, reluBMinuxAx, reluAxMinusB, Y_train, c_leafLable, treeDepth, indices_tensor, flags_tensor, batch_size=25000) # narvalxiaoyang

        # Backward pass and optimize
        objv.backward()
        optimizer.step()
        scheduler.step()


        ### barrier to ensure all ranks complete training for this epoch
        torch.distributed.barrier()

        ## Only rank 0 computes the global objvMSE_Epoch and updates the best epoch
        if rank == 0:
            objvMSE_Epoch, r2_Epoch, treeEpoch = callback.CalMSEonEpochEnd(X, Y, treeDepth, net.module.linear1, net.module.c_leafLable)

            if objvMSE_Epoch < objvMSE_EpochBest:
                objvMSE_EpochBest = objvMSE_Epoch
                r2_EpochBest = r2_Epoch
                tree_EpochBest = copy.deepcopy(treeEpoch)

    ## Return results only from rank 0
    if rank == 0:
        return objvMSE_EpochBest, r2_EpochBest, tree_EpochBest
    else:
        return None




def multiStartTreeOptbyGRAD_withC(X_rank, Y_rank, X, Y, treeDepth, indices_flags_dict, epochNum, device, warmStarts, startNum, numScale, rank, localRank):

    objvmin = 1e10
    treeOpt = None


    ############################################
    ########### if to disable the scheduler, must comment the following code; otherwise, the lr will be affected ############ 
    restartNum = 1
    # Sn = 2**restartNum-1
    Sn = restartNum
    warmupsteps = 10
    T0= math.ceil((epochNum - warmupsteps)/Sn)
    # print("T0 is {} and warmupsteps is {}".format(T0, warmupsteps))
    # epochNumNew = T0*Sn+warmupsteps
    # print("epochNumNew is {}".format(epochNumNew))
    gamma = (1/restartNum)**(1/(restartNum-1))   if restartNum != 1 else 1
    # print("gamma is {}".format(gamma))
    lrscheList = [0.01, T0, warmupsteps, gamma]
    ############################################
    

    cartWarmStart = warmStarts[0]
    treeCART = {"a": torch.tensor(cartWarmStart["a"], device=device), "b": torch.tensor(-cartWarmStart["b"], device=device), "c": torch.tensor(cartWarmStart["c"], device=device)}
    objvMSECART, r2CART = objv_cost(X, Y, treeDepth, treeCART)
    print("objvMSECART in multiStartTreeOpt: {};   r2CART: {}".format(objvMSECART, r2CART))


    # callback function
    callback = callbackFuncs()



    startNum = max(startNum, len(warmStarts))


    for idxStart in range(startNum):

        if treeDepth < 4:
            scaleMin, scaleMax = 2, 100
        elif treeDepth < 8:
            scaleMin, scaleMax = 2, 130
        elif treeDepth < 11:
            scaleMin, scaleMax = 2, 170
        else:
            scaleMin, scaleMax = 2, 200

        # nScale = numScale
        scaleList = np.logspace(np.log10(scaleMin), np.log10(scaleMax), numScale)
        print("scaleList is {}".format(scaleList))
        


        ObjvAlp = 1e10
        r2Alp = 0
        bestTreeAlp = None
        bestScale = 0


        warmStarts_cur = [warmStarts[idxStart]] if idxStart < len(warmStarts) else  [None]

    
        idxCurr = 0
        MultiAbFlag = False
        breakFlag = False  

        for scaleFactor in scaleList:
            
            if idxStart==0 and scaleFactor >=50 and ObjvAlp >= objvMSECART*0.999:
                MultiAbFlag = True
            
            if rank == 0:
                objvCurr, r2Curr, treeCurrent = treeOptbyGRADwithC(treeDepth, indices_flags_dict, epochNum, X_rank, Y_rank, X, Y, device, warmStarts_cur, scaleFactor, lrscheList, idxCurr, callback, rank, localRank,  MultiAbFlag)
                print("idx is {}; scaleFactor is {}; objvCurr is {}; r2Curr is {}".format(idxStart, scaleFactor, objvCurr, r2Curr))
            else:
                _ = treeOptbyGRADwithC(treeDepth, indices_flags_dict, epochNum, X_rank, Y_rank, X, Y, device, warmStarts_cur, scaleFactor, lrscheList, idxCurr, callback, rank, localRank,  MultiAbFlag)   

            
            ## Synchronize all ranks after training
            torch.distributed.barrier()
            
            ## Only rank 0 tracks and updates results
            if rank == 0:
                if objvCurr < ObjvAlp:
                    ObjvAlp = objvCurr
                    bestTreeAlp = copy.deepcopy(treeCurrent)
                    bestScale = scaleFactor
                if (r2Curr - r2Alp) >= 0.0005:
                    r2Alp =	r2Curr
                else:
                    if (r2Curr - r2CART) >= 0.01 and idxStart > 1:
                        # break
                        breakFlag = True
                    
                warmStarts_cur.append( {"a": treeCurrent["a"], "b": treeCurrent["b"]* (-1.0), "c": treeCurrent["c"]})                

                idxCurr += 1
            
            ## broadcast updated variables and the breakFlag to all ranks
            ## Broadcast updated variables
            if rank == 0:
                broadcast_data = {"warmStarts_cur": warmStarts_cur, "idxCurr": idxCurr, "breakFlag": breakFlag}
            else:
                broadcast_data = None

            broadcast_data = [broadcast_data]  # Wrap in a list for `broadcast_object_list`
            torch.distributed.broadcast_object_list(broadcast_data, src=0)
            
            ## Barrier to ensure all ranks have received updated variables before proceeding
            torch.distributed.barrier()

            ## Unpack broadcasted data
            broadcast_data = broadcast_data[0]
            warmStarts_cur = broadcast_data["warmStarts_cur"]
            idxCurr = broadcast_data["idxCurr"]
            breakFlag = broadcast_data["breakFlag"]


            ## Check break flag
            if breakFlag:
                break


        if rank == 0:
            if ObjvAlp < objvMSECART:
                # TreeAfterGrad = copy.deepcopy(bestTreeAlp)
                TreeAfterGrad = bestTreeAlp
            else:
                # TreeAfterGrad = copy.deepcopy(treeCART)
                TreeAfterGrad = treeCART
                ObjvAlp = objvMSECART

            if ObjvAlp < objvmin:
                objvmin = ObjvAlp
                treeOpt = copy.deepcopy(TreeAfterGrad)
            # print("objvmin is {} and objvMSECART is {}\n".format(objvmin, objvMSECART))

            print(f"bestScale={bestScale})")

    if rank == 0:
        return objvmin, treeOpt
    else:
        return None
























