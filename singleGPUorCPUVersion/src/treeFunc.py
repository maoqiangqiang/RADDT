import torch 
import sys 
import math
import h5py
import numpy as np
from typing import Dict, Tuple
import copy
from sklearn.svm import LinearSVC

sys.path.append('./src/')


# jit version of update_c 
#  tree: Dict[str, torch.Tensor] does not allow any int type value (only tensor tyoe value). 
@torch.jit.script
def update_c(X: torch.Tensor, y: torch.Tensor, treeDepth: int, tree: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    n, p = X.shape
    Tb = 2 ** treeDepth - 1  # branch node size
    Tleaf = 2 ** treeDepth  # leaf node size
    miny = torch.min(y)
    maxy = torch.max(y)

    # Initialize c and z
    c = torch.full([int(Tleaf)], (miny + maxy) / 2, device=X.device)
    z = torch.ones(n, device=X.device, dtype=torch.long)

    # Calculate the path for each data point in a vectorized manner
    for _ in range(treeDepth):
        decisions = (tree['a'][z - 1] * X).sum(dim=1) > tree['b'][z - 1] 
        z = torch.where(decisions, 2 * z + 1, 2 * z)

    z = z - (Tb + 1)
    z = z.to(torch.int64)
    unique_z, counts = torch.unique(z, return_counts=True)
    sums = torch.zeros_like(c).scatter_add_(0, z, y)
    c[unique_z] = sums[unique_z] / counts.float()
    tree['c'] = c
    return tree




@torch.jit.script
def objv_cost(X: torch.Tensor, y: torch.Tensor, treeDepth: int, tree: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    n, p = X.shape
    Tb = 2 ** treeDepth - 1
    t = torch.ones(n, device=X.device, dtype=torch.long)

    for iiii in range(treeDepth):
        # decisions = (tree['a'][t - 1] * X).sum(dim=1) >= tree['b'][t - 1]
        decisions = (tree['a'][t - 1] * X).sum(dim=1) > tree['b'][t - 1]
        t = torch.where(decisions, 2 * t + 1, 2 * t).long()

    Yhat = tree['c'][(t - (Tb + 1)).long()]
    # Manual R2 Score computation
    total_variance = torch.sum((y - torch.mean(y)) ** 2)
    residual_variance = torch.sum((Yhat - y) ** 2)
    r2_score = 1 - (residual_variance / total_variance)

    # return torch.sum((Yhat - y) ** 2)/n, r2_score
    return residual_variance/n, r2_score




## accelerate the getBranchNodes function
def getBranchNodes(ind, prelayer):
    ind -= 1
    branchNodes = [ind]
    currentNodes = [ind]
    for _ in range(prelayer-1):
        nextNodes = [2*node + j for node in currentNodes for j in [1, 2]]
        branchNodes.extend(nextNodes)
        currentNodes = nextNodes
    return branchNodes



## Tree Path Calculation and Saving 
def treePathCalculation(treeDepth, Data_device):
    branchNodeNum = 2 ** (treeDepth) - 1
    leafNodeNum = 2 ** treeDepth  

    ancestorTF_pairs = []              

    for Idx in range(leafNodeNum):
        leafIdx = Idx + branchNodeNum + 1
        log_val = math.floor(math.log2(leafIdx))
        ancestors = [leafIdx >> j for j in range(log_val, -1, -1)]
        ancestors = torch.as_tensor(ancestors, device= Data_device)
        ancestors_shifted = ancestors[1:] + 1
        oddEven = ((-1) ** ancestors_shifted + 1) / 2     # 0 -> I1, 1 -> (1-I1)
        ancestorIdxTemp = [(ancestors[ancestorIdx]-1).cpu().numpy().item() for ancestorIdx in range(len(ancestors)-1)]
        oddEvenTemp = [bool(element) for element in ((1- oddEven))]                     # 0/False -> (1-I1), 1/True -> I1

        ancestorTF_zip = list(zip(ancestorIdxTemp, oddEvenTemp))
        ancestorTF_pairs.append(ancestorTF_zip)
    ancestorTF_pairs_np = np.array(ancestorTF_pairs)
    print("ancestorTF_pairs_np: ", ancestorTF_pairs_np.shape)
    print("ancestorTF_pairs_np: ", ancestorTF_pairs_np)
    ## save the ancestorTF_pairs into a HDF5 file
    ancestorTF_File = h5py.File("./src/ancestorTF_File/ancestorTF_pairs_D"+str(treeDepth)+".hdf5", 'w')
    # Save the numpy array as a dataset in the HDF5 file
    ancestorTF_File.create_dataset("indicator_pairs", data=ancestorTF_pairs)





## Read Tree Path from HDF5 file
def readTreePath(treeDepth, device):
    ## read the treePath from the HDF5 file
    indices_flags_dict = {}

    for treeDepthEach in range(treeDepth):
        treeDepthEach += 1
        with h5py.File("./src/ancestorTF_File/ancestorTF_pairs_D"+str(treeDepthEach)+".hdf5", 'r') as ancestorTF_File:
            ancestorTF_pairs = ancestorTF_File['indicator_pairs'][:]
            ancestorTF_pairs_tensor = torch.tensor(ancestorTF_pairs, dtype=torch.long, device=device)
            ancestorTF_File.close()

        # Create tensors for indices and flags
        indices_tensor_long = ancestorTF_pairs_tensor[..., 0]
        flags_tensor_long = ancestorTF_pairs_tensor[..., 1]

        key = "D"+str(treeDepthEach)
        indices_flags_dict[key] = {
            'indices_tensor': indices_tensor_long,
            'flags_tensor': flags_tensor_long
        }

    # print(indices_flags_dict.keys())
    return indices_flags_dict






@torch.jit.script
def sampleAssignCal(X: torch.Tensor, aMatrix: torch.Tensor, bVector: torch.Tensor) -> torch.Tensor:
    n, p = X.shape
    Tb = aMatrix.shape[0]
    treeDepth = int(torch.log2(torch.tensor(Tb + 1, dtype=torch.float32)).item())
    sampleAlloc = torch.zeros((n, Tb), dtype=torch.bool, device=X.device)

    z = torch.ones(n, dtype=torch.long, device=X.device)  # Node indices
    indices = torch.arange(n, device=X.device)  # Pre-compute indices
    sampleAlloc[indices, z - 1] = True 
    
    for _ in range(treeDepth - 1):
        decisions = (aMatrix[z - 1] * X).sum(dim=1) > bVector[z - 1]
        z = torch.where(decisions, 2 * z + 1, 2 * z)
        sampleAlloc[indices, z - 1] = True

    return sampleAlloc




### compare the efficacy of adjustB and adjustBbySVM
def abjustA_b_Compr(X, aMatrix, bVector, MultiAbFlag):
    sampleAlloc = sampleAssignCal(X, aMatrix, bVector)
    Tb = aMatrix.shape[0] 
    negBVectorAdjust = torch.zeros(Tb, device=X.device, dtype=torch.float32)
    aMatrixAdjust = torch.ones_like(aMatrix)
    for t in range(Tb):

        sampleAlloc_t = sampleAlloc[:, t]
        n_t = sampleAlloc_t.sum().item()  
        if n_t <= 1:
            negBVectorAdjust[t] = -bVector[t]
            aMatrixAdjust[t, :] = aMatrix[t, :]
            continue
        X_t = X[sampleAlloc_t, :]
        ## Method 1: only abjust b with minor changes
        aVect = aMatrix[t, :]
        bt = bVector[t]
        # ax_t = torch.matmul(aVect, X_t.T) 
        ax_t = torch.matmul(X_t, aVect) 
        smallerMask = ax_t <= bt
        largerMask = ax_t >= bt
        smallerOnes = torch.where(smallerMask, ax_t, torch.full_like(ax_t, float('-inf'))).amax()
        largerOnes = torch.where(largerMask, ax_t, torch.full_like(ax_t, float('inf'))).amin()
        validSmaller = smallerOnes > float('-inf')
        validLarger = largerOnes < float('inf')
        adjustB_t = torch.where(validSmaller & validLarger, (smallerOnes + largerOnes) / 2, bt)
        negAdjustB_t = adjustB_t * (-1.0)

        ### Method 2: adjust both a and b by SVM
        pseudoY_t = torch.ones(n_t, device=X.device, dtype=torch.float32)    # pseudoY (class ) for SVC
        pseudoY_t = torch.where(smallerMask, pseudoY_t * (-1.0), pseudoY_t)

        # check if only one class in pseudoY_t
        if torch.unique(pseudoY_t).size(0) == 1:
            negBVectorAdjust[t] = negAdjustB_t
            aMatrixAdjust[t, :] = aVect
            continue

        # using SVM to adjust both a and b
        X_tCPU = X_t.cpu()
        pseudoY_tCPU = pseudoY_t.cpu()
        
        SVCModel = LinearSVC(C=1e6, max_iter=5000, dual=False)
        SVCModel = SVCModel.fit(X_tCPU, pseudoY_tCPU)
        # accuracy
        score = SVCModel.score(X_tCPU, pseudoY_tCPU)
        aVectbySVM = torch.tensor(SVCModel.coef_[0], device=X.device, dtype=torch.float32)
        negB_tbySVM = SVCModel.intercept_[0]

        norm_aVectbySVM = torch.norm(aVectbySVM, p=2)
        aVectbySVM = aVectbySVM / norm_aVectbySVM
        negB_tbySVM = negB_tbySVM / norm_aVectbySVM

        min_d_method1 = torch.min(torch.abs(ax_t + negAdjustB_t))
        ax_tbySVM = torch.matmul(X_t, aVectbySVM)
        min_d_method2 = torch.min(torch.abs(ax_tbySVM + negB_tbySVM))
        dThreshold = 0.05
        if min_d_method1 >= min_d_method2:
            negBVectorAdjust[t] = negAdjustB_t
            aMatrixAdjust[t, :] = aVect
            if MultiAbFlag == True and dThreshold > min_d_method1 > 0:
                multiFactor = dThreshold / min_d_method1
                negBVectorAdjust[t] = negAdjustB_t * multiFactor
                aMatrixAdjust[t, :] = aVect * multiFactor
        else:
            negBVectorAdjust[t] = negB_tbySVM
            aMatrixAdjust[t, :] = aVectbySVM
            if MultiAbFlag == True and dThreshold > min_d_method2 > 0:
                multiFactor = dThreshold / min_d_method2
                negBVectorAdjust[t] = negB_tbySVM * multiFactor
                aMatrixAdjust[t, :] = aVectbySVM * multiFactor

    return aMatrixAdjust, negBVectorAdjust

























if __name__ == "__main__":

    for treeDepth in [1,2,3,4,5,6,7,8,9,10,11,12]:
        Data_device = torch.device("cpu")
        treePathCalculation(treeDepth, Data_device)