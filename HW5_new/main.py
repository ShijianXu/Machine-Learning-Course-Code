# -*- coding: utf-8 -*-
"""
Created on Tue May 23 19:05:01 2017
Machine Learning
Naive Bayes
@author: XU Shijian
"""
import numpy as np
import pickle
import math

X=pickle.load(open('train_data.pkl','rb')).todense() # unsupported in Python 2
y=pickle.load(open('train_targets.pkl','rb'))
Xt=pickle.load(open('test_data.pkl','rb')).todense()

c_n = 5
f_n = 5000
d_f = 2500
c_f = 2500

D = len(y)
D_c = np.zeros(c_n)
D_c_x = np.zeros((c_n, d_f, 2))

mu = np.zeros((c_n, c_f))
sigma = np.zeros((c_n, c_f))

##trian
print("training...")
for i in range(X.shape[0]):
    y_ = int(y[i])
    D_c[y_] += 1
    
    #离散属性统计
    for j in range(d_f):
        D_c_x[y_, j, int(X[i,j])] += 1
    
    #连续属性
    mu[y_] = mu[y_] + X[i, d_f:f_n]
    

for k in range(c_n):
    mu[k] = mu[k]/D_c[k]
    
for i in range(X.shape[0]):
    y_ = y[i]    
    sigma[y_] = sigma[y_] + np.array((X[i,d_f:f_n]-mu[y_]))*np.array((X[i,d_f:f_n]-mu[y_]))

for k in range(c_n):
    sigma[k] = np.sqrt(sigma[k]/D_c[k])
    
#predict
print("predicting...")
pred = np.zeros(len(Xt))
for i in range(len(Xt)):
    if i%100 == 0:
        print(i)
        
    yt = 0
    probMax = -math.inf
    for kind in range(c_n):
        prob = 0
        prob = np.log((D_c[kind]+1)/(D+5))
        
        for j in range(d_f):
            prob += np.log((D_c_x[kind, j, int(Xt[i,j])] + 1)/(D_c[kind]+2))
        
        for j in range(c_f):
            if Xt[i,j] > 0:
                if sigma[kind,j] ==0 :
                    sigma[kind,j]=0.0001
                temp = ( 1/(np.sqrt(2*np.pi)*sigma[kind,j]) ) *np.exp(- (Xt[i,j]-mu[kind,j])*(Xt[i,j]-mu[kind,j])/(2*sigma[kind,j]*sigma[kind,j]))
                if temp != 0:
                    prob += np.log(temp)
                else:
                    prob+=0
        
        if prob > probMax:
            probMax = prob
            yt = kind
        else:
            pass
        
    pred[i]=yt
        
np.savetxt('test_predictions.csv', pred, fmt="%d", delimiter=",")