# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 18:50:30 2017

@author: Xu Shijian 141220120
"""

import numpy as np
from sklearn.model_selection import KFold
import csv

def sigmoid(num):
    return 1.0 /(1+np.exp(-num))

def calcHessian(data, prob):
    hessianMatrix = []
    n = len(data)

    for i in range(n):
        row = []
        for j in range(n):
            row.append(-data[i]*data[j]*(1-prob)*prob)
        hessianMatrix.append(row)
    return hessianMatrix

def newtonMethod(train_x, train_y, iterNum=10):
    m = len(train_x)
    n = len(train_x[0])
    beta = [0.0] * n #初值很重要，如果初值为1，则精度很低

    while(iterNum):
        gradientSum = [0.0] * n
        hessianMatSum = [[0.0] * n]*n
        for i in range(m):
            prob = sigmoid(np.dot(train_x[i], beta))
            gradient = np.dot(train_x[i], (train_y[i] - prob)/m)
            gradientSum = np.add(gradientSum, gradient)
            hessian = calcHessian(train_x[i], prob/m)
            for j in range(n):
                hessianMatSum[j] = np.add(hessianMatSum[j], hessian[j])
        try:
            hessianMatInv = np.linalg.inv(hessianMatSum)
        except:
            continue
        for k in range(n):
            beta[k] -= np.dot(hessianMatInv[k], gradientSum)

        iterNum -= 1
    return beta


def do_test(beta, testdata, testindex, count):
    csvFile = open("fold%d.csv"%count, 'w')
    writer = csv.writer(csvFile,lineterminator="\n")
    row = []
    
    for i in range(len(testdata)):
        x=testdata[i]
        prod = np.dot(beta, x)
        prob = sigmoid(prod)
        if prob> 0.5:
            row.append((testindex[i]+1,1))
        else:
            row.append((testindex[i]+1,0))

    writer.writerows(row)
    csvFile.close()

target = np.genfromtxt('targets.csv')
data = np.genfromtxt('data.csv',delimiter=',', dtype=np.float)

kf = KFold(n_splits=10)
kf.get_n_splits(target)
count = 0
for train, test in kf.split(target):
    count +=1
    testdata=[]
    train_x=[]
    train_y=[]
    print(train, test)
    for index in train:
        x=data[index]
        y=target[index]
        train_x.append(x)
        train_y.append(y)

    for index in test:
        x=data[index]
        testdata.append(x)
    
    #now we have train data(x,y) and test data(x)
    #after the 10 fold cv, we need to calculate an average accuracy.
    #next, use the train data to generate beta by Newton's Method
    newtonMethod(train_x, train_y, 10)
    beta = newtonMethod(train_x, train_y, 10)
    
    do_test(beta, testdata, test, count)
    