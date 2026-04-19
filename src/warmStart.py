from sklearn.tree import _tree
from sklearn import tree
import numpy as np
import torch 
import sklearn.metrics as metrics


def getNodesId2(ind, Hind):
    ind -= 1                  # index starts from 0
    branchNodes = [ind]
    currentNodes = [ind]
    for _ in range(Hind-1):
        nextNodes = [2*node + j for node in currentNodes for j in [1, 2]]
        branchNodes.extend(nextNodes)
        currentNodes = nextNodes
    leftLeaf = 2*(ind+1) if Hind == 1 else 2*(nextNodes[0]+1)      
    leafNodes = [2 * (ind+1) + i  for i in [0, 1]] if Hind ==1 else [2 * (leafNodeParent+1) + i for leafNodeParent in nextNodes for i in [0, 1]] 
    return branchNodes, leftLeaf, leafNodes


## retrieve the parameters abc of the trained tree model
def regTreeWarmStart(model, treeDepth, p):
    tree_ = model.tree_                  
    branchNode_inputDepth = 2**(treeDepth) - 1
    # Fitted_treeDepth = model.get_depth()
    # print("Fitted_treeDepth: ", Fitted_treeDepth)
    leafNode_inputDepth = 2**(treeDepth)
    a = np.random.randint(p, size=branchNode_inputDepth  )    
    b = np.random.rand(branchNode_inputDepth)* (2.0)+(-1.0)     
    c = [0.5]*leafNode_inputDepth

    ab0indList = []     
    def warmStartPara(node, ind):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            featureIdx = tree_.feature[node]
            threshold = tree_.threshold[node]
            a[ind-1] = featureIdx
            b[ind-1] = -threshold
            
            node_l = 2 * ind
            node_r = 2 * ind + 1
            warmStartPara(tree_.children_left[node], node_l)
            warmStartPara(tree_.children_right[node], node_r)
        
        else:

            if ind <= branchNode_inputDepth:
                currDepthForInd = int(np.log2(ind))
                diffDepthInbd = treeDepth - currDepthForInd
                ab0NodeListForEachInd, leftLeaf, leafNodesFor0 = getNodesId2(ind, diffDepthInbd)
                ab0indList.extend(ab0NodeListForEachInd)
                for eachLeafNodes in leafNodesFor0:
                    c[eachLeafNodes-1-branchNode_inputDepth] = tree_.value[node].squeeze()

            else:
                c[ind-1-branchNode_inputDepth] = tree_.value[node].squeeze()

    warmStartPara(0, 1)
    return a, b, c, ab0indList


def CARTRegWarmStart(X, Y, treeDepth, device):
    model = tree.DecisionTreeRegressor(max_depth=treeDepth, min_samples_leaf=1, random_state=0)
    if device == torch.device('cpu'):
        X_np, Y_np = X, Y
    else:
        X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()

    if X_np.shape[0] < 1:
        branchNodeNum = 2**(treeDepth) - 1
        leafNodeNum = 2**(treeDepth)
        p = X_np.shape[1]
        b = [-0.5]*branchNodeNum
        c = [0.5]*leafNodeNum
        a = np.zeros((branchNodeNum, p), dtype="float32")
        a[:,0] = 1
        return a, np.asarray(b,dtype="float32"), np.asarray(c, dtype="float32")

    else:
        model = model.fit(X_np, Y_np)
        p = X.shape[1]
        a, b, c, ab0indList = regTreeWarmStart(model,treeDepth, p)
        a_all = np.eye(p, dtype="float32")[a]       

        return a_all, np.asarray(b,dtype="float32"), np.asarray(c, dtype="float32")



