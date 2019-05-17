#   test k-Nearest Neighbours method
#   by Qiyang Ma, from May 6, 2019 to May 12, 2019

import numpy as np
import operator

def createDataSet():
#   create dataset for testing k-Nearest Neighbor method
    group = np.array([[1.0,1.1],[1.0,1.1],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0( inx, dataSet, labels, k ):
#   calculate the distance between Inx and every points in the dataSet, choose k labels corresponded to k mi-    nimum values and get the label     of Inx.
#   Inx: object point, for get the label corresponds to Inx point.
#   dataSet: sample dataset.
#   labels: the labels correspond to every points in the dataset.
#   k: screen out k minimum values
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inx, (dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDist_ind = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabels = labels[sortedDist_ind[i]]
        classCount[voteLabels] = classCount.get(voteLabels,0)+1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix( filename ):
    fr = open(filename)
    datalines  = fr.readlines()
    nlength = len(datalines)
    Mat = np.zeros((nlength,3))
    MatLabels = []
    n = 0
    for line in datalines:
#        linestr = line.strip()
#        linespl = linestr.split('\t') 
        liness = line.split()
        Mat[n,:] = liness[0:3]
        MatLabels.append(int(liness[3]))
        n += 1
    return Mat,MatLabels

def normalize(DataSet,opt):
#   normalize dataset according to the rows or columns
#   Input:    opt: 1 - rows; 0 - columns
#   Output:
#   DataNew: the normalized dataset;
#   dataRange: the range of every rows or columns;    DSmin: the minimum of every rows or columns
    DSmax = DataSet.max(opt)
    DSmin = DataSet.min(opt)
    dataRange = DSmax-DSmin
    DataNew = np.zeros(DataSet.shape)
    m = DataSet.shape[0]
    DataNew = (DataSet-np.tile(DSmin,(m,1)))/np.tile(dataRange,(m,1))
    return DataNew, dataRange, DSmin

import random as rd

def accuracyTest( filename, Ratio, k ):
#   calculate error rate of kNN method
#   Input:
#   filename: the file of sample dataset
#   Ratio: Ratio*len(dataset) samples in sample dataset were be randomly selected.
#   Output:
#   AccuracyRatio: the accuracy ratio of the kNN method
    mat,matLabels = file2matrix( filename )
    matNew = normalize(mat,0)
    num = len(matNew)
    bgnum = range(num)
    nTestList = sorted(rd.sample(range(num),int(Ratio*num)),reverse = True)
    bgLabels = matLabels[:]
    errorCount = 0.0
    for m in nTestList:
        del bgLabels[m]
        bgnum.remove(m)
    nTestList = sorted(nTestList)
    for n in nTestList:
        clsResult = classify0(matNew[n,:],matNew[bgnum,:],bgLabels,k)
        print 'No. %d, %d, the result is %d, the real is %d' % (n+1,mat[n,0],clsResult,matLabels[n])
        if (clsResult != matLabels[n]): errorCount += 1.0 
    print 'the error rate is %f' % (errorCount/(Ratio*num))
    print errorCount, Ratio, num
    AccuracyRatio = 1-errorCount/(Ratio*num)
    return AccuracyRatio

def classifyPerson( filename ):
#   final function. classify different person to 'not at all', 'in small does' and 'in large does'
#   output: the classification of the objective person
    flyM = float(raw_input('Fly Miles: '))
    vedioT = float(raw_input('Vedio Time:'))
    icecreamV = float(raw_input('icecream Volumn:'))
    targetList = [ flyM, vedioT, icecreamV ]
    samples,sampleLabels = file2matrix( filename )
    samples,spRange,spMin = normalize( samples,0 )
    classifyResult = classify0((targetList-spMin)/spRange,samples,sampleLabels,3)
    result = [ 'not at all', 'in small does', 'in large does']
    print 'The person: ',result[classifyResult-1]
