# Decision Trees Method
# for testing decision trees method in Machine Learning in Action.
# edit by Qiyang Ma, from May 16, 2019 to May 22, 2019

import numpy as np
import operator

def createDataSet():
#   create Dataset
#   Output: 
#   featureValue: dataset
#   featureLabels: labels list of all the features
    featureValue = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    featureLabels = ['no surfacing','flippers']
    return featureValue, featureLabels
#for test
#   temp = [[1,'yes',1,1],[1,'yes',1,1],[1,'no',0,0],[0,'no',1,0],[0,'no',1,0]]
#   tempLabels = ['no surfacing','flippers','test']
#   return temp,tempLabels

from math import log

def calcSNentropy( dataSet, k=-1 ):
#   calculate the Shannon Entropy of the k column of the dataset, generally speaking, k column value represents ultimate goal of classify
#   Input:
#   k is target column index, which the data of this column represents ultimate goal of classify. default is     -1, means the last column is object classication
#   Output: SNentropy: the Shannon entropy of the dataset 
    length = len(dataSet)
    labelCounts = {}
    for dataline in dataSet:
        newValue = dataline[k]
        if newValue not in labelCounts.keys():
            labelCounts[newValue] = 0.0
        labelCounts[newValue] += 1
    SNentropy = 0.0
    for i in labelCounts.values():
        prob = i/length
        SNentropy -= prob*log(prob,2)
    return SNentropy

def createSubDataSet( dataSet, ftIndex, ftValue ):
#   Create subset composed with the dataSet[:,ftIndex] = ftValue row and delete the ftIndex column.
#   Input:
#   ftIndex: the index of the feature column
#   ftValue: feature value, using to split dataset.
    subDataSet = []
    for dataline in dataSet:
        if dataline[ftIndex] == ftValue:
            lineTemp = dataline[:ftIndex]
            lineTemp.extend(dataline[ftIndex+1:])
            subDataSet.append(lineTemp)
    return subDataSet

def chooseBestFeature( dataSet, k=-1 ):
#   choose best feature column in dataset, delete dataset[:,k], which this column represents class of data
#   Output:   
#   bestFeature: the Index or the column number of the best feature
    numFeature = len(dataSet[0])
    numList = range(numFeature)
    del numList[k]
    bestFeature = -1
    bestInfoGain = 0.0
    newSNentropyList = []
    baseSNentropy = calcSNentropy(dataSet,k)
    for i in numList:
        featureValues = [values[i] for values in dataSet]
        featuresList = set(featureValues)
        newSNentropy = 0.0
        for ftValue in featuresList:
            subDataSet = createSubDataSet(dataSet,i,ftValue)
            prob = float(len(subDataSet))/len(dataSet)
            if i <= k: k = k-1
            newSNentropy += prob*calcSNentropy(subDataSet,k)
        infoGain = baseSNentropy-newSNentropy
        newSNentropyList.append(newSNentropy)
        if infoGain>bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityClass( classList ):
#   when it comes to the last feature, the object classifaction result is not unique, select catagory owning the largest number as result.
#   Input: classList: the rest classification list after go through all the feature
#   Output: sortedClassCount[0][0]: the catagory owning the largest number
    classCount = {}
    for result in classList:
        if result not in classCount.keys():
            classCount[result] = 0
        classCount[result] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
    
def createTree( dataSet, dataLabels, k=-1 ):
#   the main function 1. create the decision tree.
    classList = [examples[k] for examples in dataSet]
    if classList.count(classList[0]) == len(classList): return classList[0]
    if len(dataSet[0]) == 1: return majorityClass(classList)
    bestFeatIndex = chooseBestFeature(dataSet,k)
    if bestFeatIndex <= k and k != -1: k = k-1
    dataLabeltemps = dataLabels[:]    # keep complete dataLabels: after del variable, original list no change
    bestFeat = dataLabeltemps[bestFeatIndex]
    del dataLabeltemps[bestFeatIndex]
    myTree = {bestFeat:{}}
    featValue = [examples[bestFeatIndex] for examples in dataSet]
    uniFeatValue = set(featValue)
    for Value in uniFeatValue:
        subLabels = dataLabeltemps[:]
        subDataSet = createSubDataSet(dataSet,bestFeatIndex,Value)
        myTree[bestFeat][Value] = createTree(subDataSet,subLabels,k)
    return myTree
    
def classifyTree( tree, featLabels, testValue ):
#   the main function 2. get the classification result of testValue.
#   Input:
#   featLabels: Labels list of all the features.
#   testValue: test Value, for finding the classification of testValue.
#   Output: classification result of testValue.
    feature = list(tree.keys())[0]
    branchTree = tree[feature]
    ind = featLabels.index(feature)
    for key in branchTree.keys():
        if str(testValue[ind]) == str(key):
            if type(branchTree[key]) is dict: classifyTree( branchTree[key],featLabels,testValue )
            else: classifyResult = branchTree[key]
    return classifyResult  

import pickle
#   store and grab tree in file.

def storeTree( tree, filename ):
    fw = open(filename,'w')
    pickle.dump(tree,fw)
    fw.close()

def grabTree( filename ):
    fr = open(filename)
    return pickle.load(fr)
