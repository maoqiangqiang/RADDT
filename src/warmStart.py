from sklearn.tree import _tree
from sklearn import tree
import numpy as np
import torch 
import sklearn.metrics as metrics

def onehot_coding(target, device, output_dim):
    """Convert the class labels into one-hot encoded vectors."""
    target_onehot = torch.FloatTensor(target.size()[0], output_dim).to(device)
    target_onehot.data.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1.0)
    return target_onehot


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
def clfTreeWarmStart(model, treeDepth, p):
    tree_ = model.tree_                  
    branchNode_inputDepth = 2**(treeDepth) - 1
    # Fitted_treeDepth = model.get_depth()
    # print("Fitted_treeDepth: ", Fitted_treeDepth)
    leafNode_inputDepth = 2**(treeDepth)
    a = np.random.randint(p, size=branchNode_inputDepth  )    
    b = np.random.rand(branchNode_inputDepth)* (2.0)+(-1.0)    
    c = [0]*leafNode_inputDepth

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
                    treeNodeValue = tree_.value[node].squeeze()
                    treeNodeLabel = np.argmax(treeNodeValue)
                    c[eachLeafNodes-1-branchNode_inputDepth] = treeNodeLabel
            else:
                ctreeNodeValue = tree_.value[node].squeeze()
                ctreeNodeLabel = np.argmax(ctreeNodeValue)
                c[ind-1-branchNode_inputDepth] = ctreeNodeLabel

    warmStartPara(0, 1)
    return a, b, c, ab0indList


def CARTClfWarmStart(X, Y, treeDepth, nClass,  device):
    # classification
    model = tree.DecisionTreeClassifier(max_depth=treeDepth, min_samples_leaf=1, random_state=0)
    if device == torch.device('cuda'):
        X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    else:
        X_np, Y_np = X, Y

    if X_np.shape[0] < 1:
        branchNodeNum = 2**(treeDepth) - 1
        leafNodeNum = 2**(treeDepth)
        p = X_np.shape[1]
        b = [-0.5]*branchNodeNum
        c = [0]*leafNodeNum
        a = np.zeros((branchNodeNum, p), dtype="float32")
        a[:,0] = 1

        c_np = np.asarray(c, dtype="int64")
        cInitTensor = torch.tensor(c_np, dtype=torch.long)
        c_onehot = onehot_coding(cInitTensor, "cpu", nClass)
        return a, np.asarray(b,dtype="float32"), c_onehot

    else:
        model = model.fit(X_np, Y_np)
        p = X.shape[1]
        a, b, c, ab0indList = clfTreeWarmStart(model,treeDepth, p)
        a_all = np.eye(p, dtype="float32")[a]       
        c_np = np.asarray(c, dtype="int64")
        cInitTensor = torch.tensor(c_np, dtype=torch.long)
        c_onehot = onehot_coding(cInitTensor, "cpu", nClass)

        return a_all, np.asarray(b,dtype="float32"), c_onehot



