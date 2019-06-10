#!/bin/python3
# Naive Bayes method, version: 3.4.9
# for testing Naive Bayes method in Machine Learning in Action.
# edit by Qiyang Ma, from June 3, 2019 to June 10, 2019

import numpy as np

def loadWords():
    wordList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classList = [1,1,0,1,0,1]    #1 is abusive, 0 not
    return wordList,classList

def setofWords( wordList ):
#   create a set including all the words. The words are all unique.
#   Inputs: wordList: word lists from sample texts.
#   Outputs: wordSet: a word set including all the words from texts.
    wordSet = set([])
    for line in wordList:
        wordSet = wordSet | set(line)
    return list(wordSet)

def setofWordsVec( wordSet, inputWords ):
#   word set transform to token vector, only include 0 and 1.
#   Inputs: inputWords: testing words.
#   Outputs: wordVec: token vector transformed from word list. 0 - no contain, 1 - contain.    
    wordVec = [0]*len(wordSet)
    for x in inputWords:
        if x in wordSet: wordVec[wordSet.index(x)] = 1
    return wordVec

def bagofWordsVec( wordSet, inputWords ):
#   word set transform to token vector, the number equals the number of the word.
    wordVec = [0]*len(wordSet)
    for x in inputWords:
        if x in wordSet: wordVec[wordSet.index(x)] += 1
    return wordVec

def list2Mat( wordList, freqOpt=0, opt=0 ):
#   word list to token vectors, then form token matrix (type: list).
#   Input:
#   freqOpt: option of delete top k words in the word set. 0 - do not delete; other - delete.
#   opt: option of set or bag. 0 - set of word vectors; other - bag of word vectors
    wordSet = setofWords( wordList )
    if freqOpt != 0:
        freqWords = frequentWords(wordSet,WordList)
        for word in freqWords:
            if word[0] in wordSet: wordSet.remove(word[0])
    wordMat = []
    if opt == 0:
        for i in range(len(wordList)): wordMat.append(setofWordsVec( wordSet,wordList[i] ))
    else:
        for i in range(len(wordList)): wordMat.append(bagofWordsVec( wordSet,wordList[i] ))
    return wordMat

import operator
def frequentWords( wordSet,wordList,k=30 ):
    fullText = [];
    for words in wordList: fullText = fullText.attend(words)
    wordsDic = {};
    for word in wordSet: wordsDic[word] = fullText.count(word)
    sortedWords = sorted(wordsDic.items(),key=operator.itemgetter(1),reverse=True)
    return sortedWords[:k] 

def calProbability( wordMat, classList ):
#   calculate probabilities from token matrix transformed from word lists.
#   Input: 
#   wordMat: token matrix transformed from word lists.
#   classList: classification of word lists, acquired from sample text.
    num = len(classList)
    classResult = list(set(classList))
    pClass = {}; pVec = {}; pSum =  {}
    for result in classResult:
        pClass[result] = classList.count(result)/float(num)
        pVec[result] = np.ones(len(wordMat[0]))
        pSum[result] = 2.0
    for i in range(num):
        pVec[classList[i]] += wordMat[i]; pSum[classList[i]] += sum(wordMat[i])
    for result in classResult:
        pVec[result] = np.log(pVec[result]/pSum[result])
    return pVec, pClass

#   eg. inputWords = ['stupid','my']
inputWords = ['stupid','my']

def classifyNB( wordList, classList, inputWords=inputWords, freqopt=0, opt=0 ):
#   Main function. Wrap up every function above to classify inputWords (type: list).
#   Input:
#   WordList: word samples; classList: the classification of word lists. both of them acquired from texts.
#   inputWords: objective words which are used to classify.
#   freqOpt: option of delete top k words in the word set. 0 - do not delete; other - delete.
#   opt: option of set or bag. 1 - set of word vectors; other - bag of word vectors
    wordMat = list2Mat( wordList,freqOpt=0,opt=0 )
    pVec,pClass = calProbability( wordMat,classList )
    if opt == 1: wordVec = setofWordsVec( setofWords(wordList),inputWords )
    else: wordVec = bagofWordsVec( setofWords(wordList),inputWords )
    p = {};
    for key in list(pClass.keys()):
        p[key] = sum( wordVec * pVec[key] ) + np.log(pClass[key])
    sortedClass = sorted(p.items(),key=operator.itemgetter(1),reverse=True)
    print('{0} classified as: {1:2d}'.format( inputWords,sortedClass[0][0]))
    return sortedClass[0][0]

import re
def textRead( fileName ):
#   read a list of word from the text document.
#   Input: fileName: the text document. Output: words list in the text document.
    string = open(fileName).read()
    reg = re.compile(r'\W+')
    wordList = reg.split(string)
    return [word.lower() for word in wordList if len(word)>3] # <3 means delete useless short words.

import random as rd
def emailFilterNB():
#   Example function. Using classifyNB function to classify email into spam and ham (type: text). An naive bayes method application.
#   create the lists of words from email text documents--------
    wordList = []; classList = []
    for i in range(1,26):
        words = textRead('email/ham/{0:d}.txt'.format(i))
        wordList.append(words); classList.append(0)
        words = textRead('email/spam/{0:d}.txt'.format(i))
        wordList.append(words); classList.append(1)
#   -------------------------------------------------------------
    trainIndex = list(range(50));testIndex = sorted(rd.sample(range(50),10))
    for x in testIndex: trainIndex.remove(x)
    trainList = []; trainClassList = []
    for ind in trainIndex:
        trainList.append(wordList[ind])
        trainClassList.append(classList[ind])
    errorCount = 0
    for ind in testIndex:
        inputWords = wordList[ind]
        classResult = classifyNB( trainList,trainClassList,inputWords )
        if classResult != classList[ind]: errorCount += 1
    print('The error rate is: {0:.2f}'.format(errorCount/10))
