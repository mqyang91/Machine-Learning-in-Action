# plot the decision tree by matplotlib, version: 3.4.9
# for learn how to use matplotlib for plotting the decision tree in Machine Learning in Action
# edit by Qiyang Ma. from May 25, 2019 to 

# example: tree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
# notice: because both feature label and classification value are simultaneous, so the number of big parentheses must be even in the desicion trees 

import matplotlib.pyplot as plt

def getNumLeaf( tree ):
# the number of leaves of the decision tree, in order to design the structure(X-axis) of the decision tree
    branchLabel = list(tree.keys())[0]
    branchTree = tree[branchLabel]
    numLeafs = 0
    for key in branchTree.keys():
        if type(branchTree[key]) is dict: numLeafs += getNumLeaf( branchTree[key] )  
        else: numLeafs += 1
    return numLeafs

def getTreeDepth( tree ):
# the depth of the decision tree, in order to design the structure(Y-axis) of the decision tree
    branchLabel = list(tree.keys())[0]
    branchTree = tree[branchLabel]
    depth = 0
    for key in branchTree.keys():
        if type(branchTree[key]) is dict: 
            thisDepth = 1+getTreeDepth( branchTree[key] )
        else: thisDepth = 1
        if thisDepth > depth: depth = thisDepth
    return depth

def picTree( tree ):
# the main function. plot the decision tree figure
# Output: draw the tree picture
    fig = plt.figure()
    picTree.ax = plt.subplot(111,frameon=False)
    plt.xticks([]);plt.yticks([])
    picTree.treeW = getNumLeaf( tree ) 
    picTree.treeD = getTreeDepth( tree )
    picTree.xoff = -1/2/picTree.treeW; picTree.yoff = 1.0;
    parentPt = (0.5,1)
    plotTree( tree,parentPt,'' )
    plt.show()

nodeStyle = dict(boxstyle='sawtooth',fc='0.8');leafStyle = dict(boxstyle='round4',fc='0.8')
arrowStyle = dict(arrowstyle='<-')

def plotTree( tree,parentPt,featVar ):
# plot nodes, leaves and line of the tree by using recursion 
    branchLabel = list(tree.keys())[0]
    childPt = (picTree.xoff+1/2/picTree.treeW+getNumLeaf(tree)/2/picTree.treeW,picTree.yoff)
    plotFeatVar(featVar,parentPt,childPt)
    plotNode( branchLabel,parentPt,childPt,nodeStyle )
    branchTree = tree[branchLabel]
    picTree.yoff = picTree.yoff-1/picTree.treeD
    for key in branchTree.keys():
        if type(branchTree[key]) is dict:
            plotTree(branchTree[key],childPt,key )
        else:
            picTree.xoff = picTree.xoff+1/picTree.treeW
            nodePt = (picTree.xoff,picTree.yoff)
            plotFeatVar(key,childPt,nodePt)
            plotNode( branchTree[key],childPt,nodePt,leafStyle )
    picTree.yoff = picTree.yoff+1/picTree.treeD

def plotFeatVar( featVar, parentPt, childPt ):
# text feature variate results
    xMid = (parentPt[0]+childPt[0])/2
    yMid = (parentPt[1]+childPt[1])/2
#    print(xMid,yMid,featVar)
#    picTree.ax.text(xMid,yMid,featVar)
    plt.text(xMid,yMid,featVar)

def plotNode( nodeText, parentPt, nodePt, boxStyle ):
# annotate in the node or leaves
    picTree.ax.annotate(nodeText,xy=parentPt,xycoords='axes fraction',xytext=nodePt,textcoords='axes fraction',va='center',ha='center',bbox=boxStyle,arrowprops=arrowStyle)
