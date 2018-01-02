# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:26:09 2017

@author: XU Shijian 141220120
"""

import numpy as np
import csv

train_data = np.genfromtxt('train_data.csv',delimiter=',')
train_targets = np.genfromtxt('train_targets.csv')

inputNum = 400
hiddenNum = 100
outputNum = 10

#weight-1 from input layer to hidden layer
weight1 = np.random.uniform(-0.05,0.05,(inputNum,hiddenNum))
#weight-2 from hidden layer to output layer
weight2 = np.random.uniform(-0.05,0.05,(hiddenNum,outputNum))
#theta-1 hidden layer bias
theta1 = np.zeros(hiddenNum)
#theta-2 output layer bias
theta2 = np.zeros(outputNum)
        
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def arr(x):
    y = np.zeros(10)
    return y

def Train(train_data, train_targets, weight1, weight2, theta1, theta2, hiddenNum,outputNum,learn_rate=0.1, epochs=20):
    for nn in range(epochs):
        print(nn)
        for k in range(train_data.shape[0]):
            y=arr(train_targets[k])
            y[train_targets[k]]=1
            
            #####################################
            #   计算输出
            x = train_data[k]
            b = np.dot(x,weight1)-theta1
            b = sigmoid(b)  #100*1
            yhat = np.dot(b, weight2)-theta2
            yhat = sigmoid(yhat)
            #####################################
            #   误差后向传播
            out_g=yhat*(1-yhat)*(y-yhat) #10*1
            hidden_e=b*(1-b)*np.dot(weight2,out_g) #100*1
            
            for i in range(0, outputNum):
                weight2[:,i] += learn_rate * out_g[i] * b
            
            for i in range(0, hiddenNum):
                weight1[:,i] += learn_rate * hidden_e[i] * x
            
            theta1 -=learn_rate*hidden_e
            theta2 -=learn_rate*out_g
     

learn_rate = 0.1
epochs = 60
Train(train_data, train_targets, weight1, weight2, theta1, theta2,hiddenNum,outputNum,learn_rate, epochs)

csvFile = open("test_predictions.csv", 'w')
writer = csv.writer(csvFile,lineterminator="\n")

test_data = np.genfromtxt('test_data.csv',delimiter=',')
test_targets = np.genfromtxt('test_targets.csv')
ans = 0
row = []
for count in range(len(test_data)):
    b = np.dot(test_data[count], weight1) - theta1
    b = sigmoid(b)
    yhat = np.dot(b, weight2) - theta2
    yhat = sigmoid(yhat)
    row.append((np.argmax(yhat),))
writer.writerows(row)
csvFile.close()