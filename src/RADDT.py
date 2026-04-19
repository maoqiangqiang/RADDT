import torch 
import copy 
import math
import sys 
sys.path.append('./src/')

from warmStart import CARTRegWarmStart
from treeFunc import objv_cost, update_c, abjustA_b_Compr

from modifiedScheduler import ChainedScheduler
import numpy as np

import os 
##  enforce deterministic behavior in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

# from torch.utils.tensorboard import SummaryWriter




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
        self.relu = torch.nn.ReLU()
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
            a_grad = netLinear1.weight.clone().detach()  # Ensuring no gradients
            b_grad = netLinear1.bias.clone().detach() * (-1.0)
            c_grad = net_c_leafLable.clone().detach()  # Ensuring no gradients
            treeEpoch = {"a": a_grad, "b": b_grad, "c": c_grad}
            
        treeEpoch = update_c(X_train, Y_train, treeDepth, treeEpoch)
        objvMSE_Epoch, r2_Epoch = objv_cost(X_train, Y_train, treeDepth, treeEpoch)
        return objvMSE_Epoch, r2_Epoch, treeEpoch
        



def treeOptbyGRADwithC(treeDepth, indices_flags_dict, epochNum, X_train, Y_train, device, warm_starts, scaleFactor, lrscheList,  idxStart, callback, adjust_ab=False):

    ## hyperparameters
    learningRate, T0, warmupsteps, gamma = lrscheList[0], lrscheList[1], lrscheList[2], lrscheList[3]


    ##  net
    p = X_train.shape[1]
    scale = torch.tensor([scaleFactor], device=device)
    net_placeholder = branchNodeNet(treeDepth, p, scale, None, None, None).to(device, non_blocking=True)

    ### initialize weight and bias of net
    a, b, c = initialize_parameters(idxStart, net_placeholder, warm_starts, X_train, Y_train, device, adjust_ab)

    net = branchNodeNet(treeDepth, p, scale, a, b, c).to(device, non_blocking=True)


    aInit = copy.deepcopy(a)
    bInit = copy.deepcopy(b)
    cInit = copy.deepcopy(c)

    ## Optimizer 
    optimizer = torch.optim.AdamW(net.parameters(), lr=learningRate)

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


    objvMSE_EpochBest = float('inf')
    r2_EpochBest = float('-inf')
    tree_EpochBest = None

    for epoch in range(epochNum):

        optimizer.zero_grad(set_to_none=True)   

        # Forward pass with a batch
        reluBMinuxAx, reluAxMinusB, c_leafLable  = net(X_train)
        objv = objectiveFuncwithC(scaleFactor, reluBMinuxAx, reluAxMinusB, Y_train, c_leafLable, treeDepth, indices_tensor, flags_tensor, batch_size=17000)


        ## check the real mse loss of the current tree
        objvMSE_Epoch, r2_Epoch, treeEpoch = callback.CalMSEonEpochEnd(X_train, Y_train, treeDepth, net.linear1, net.c_leafLable)
        if objvMSE_Epoch < objvMSE_EpochBest:
            objvMSE_EpochBest = objvMSE_Epoch
            r2_EpochBest = r2_Epoch
            tree_EpochBest = copy.deepcopy(treeEpoch)

        # Backward pass and optimize
        objv.backward()


        optimizer.step()
        scheduler.step()

        
    return objvMSE_EpochBest, r2_EpochBest, tree_EpochBest, aInit, bInit, cInit







def multiStartTreeOptbyGRAD_withC(X_train, Y_train, treeDepth, indices_flags_dict, epochNum, device, warmStarts, startNum, numScale):

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
    objvMSECART, r2CART = objv_cost(X_train, Y_train, treeDepth, treeCART)
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

        scaleList = np.logspace(np.log10(scaleMin), np.log10(scaleMax), numScale)
        print("scaleList is {}".format(scaleList))



        objvAlp = 1e10
        r2Alp = 0
        bestTreeAlp = None
        bestScale = 0


        warmStarts_cur = [warmStarts[idxStart]] if idxStart < len(warmStarts) else  [None]

    
        idxCurr = 0

        MultiAbFlag = False
        for scaleFactor in scaleList:

            
            if scaleFactor >= 50:
                if idxStart == 0 and objvAlp >= objvMSECART * 0.999:
                    MultiAbFlag = True
                elif idxStart != 0 and objvAlp >= objvRandInit * 0.999:
                    MultiAbFlag = True


            objvCurr, r2Curr, treeCurrent, aInit, bInit, cInit = treeOptbyGRADwithC(treeDepth, indices_flags_dict, epochNum, X_train, Y_train, device, warmStarts_cur, scaleFactor, lrscheList, idxCurr, callback, MultiAbFlag)
            
            print("idx is {}; scaleFactor is {}; objvCurr is {}; r2Curr is {}".format(idxStart, scaleFactor, objvCurr, r2Curr))

            if idxStart != 0 and scaleFactor == scaleList[0]:
                objvRandInit, r2RandInit = objv_cost(X_train, Y_train, treeDepth, {"a": aInit, "b": bInit, "c": cInit})


            if objvCurr < objvAlp:
                objvAlp = objvCurr
                bestTreeAlp = copy.deepcopy(treeCurrent)
                bestScale = scaleFactor
               
            if (r2Curr - r2Alp) >= 0.0005:
                r2Alp =	r2Curr
            else:
                if (r2Curr - r2CART) >= 0.01 and idxStart > 1:
                    break
                
            warmStarts_cur.append( {"a": treeCurrent["a"], "b": treeCurrent["b"]* (-1.0), "c": treeCurrent["c"]})                

            idxCurr += 1



        if objvAlp < objvMSECART:
            TreeAfterGrad = bestTreeAlp
        else:
            TreeAfterGrad = treeCART
            objvAlp = objvMSECART
        
        if objvAlp < objvmin:
            objvmin = objvAlp
            treeOpt = copy.deepcopy(TreeAfterGrad)

        print("bestScale is {}; \n".format(bestScale))
    
    return objvmin, treeOpt
























