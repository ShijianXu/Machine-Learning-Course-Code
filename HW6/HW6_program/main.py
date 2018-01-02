# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 19:29:17 2017
AdaBoost
@author: Xu Shi-Jian
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import copy

X=np.genfromtxt('data.csv',delimiter=',')
y=np.genfromtxt('targets.csv')

class AdaBoost(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.base_model = LogisticRegression(penalty = 'l2', C=1)
        
    def fit(self, x_train, y_train, base_num):
        baseClassifier=[]
        alphaCollect = []
        m = len(y_train)
        self.D = np.ones(m)/m
        
        for k in range(base_num):
            #base = LogisticRegression(C=50)
            base = self.base_model.fit(x_train, y_train, self.D)
            base.fit(x_train, y_train, self.D)
            
            
            pred = base.predict(x_train)
            errArr = np.ones(m)
            
            for j in range(m):
                if pred[j]==y_train[j]:
                    errArr[j] = 0
                          
            errRate = sum(pred != y_train) / m
                         
            if errRate > 0.5:
                return baseClassifier,alphaCollect
            
            if errRate == 0.0:
                errRate+=1/m
            alpha = 0.5*np.log((1.0-errRate)/errRate)
            baseClassifier.append(copy.deepcopy(base))
            alphaCollect.append(alpha)
            
            for i in range(m):
                if errArr[i]==0:
                    self.D[i] = self.D[i]*np.exp(-1.0*alpha)
                else:
                    self.D[i] = self.D[i]*np.exp(alpha)
            
            self.D = self.D/sum(self.D)
            #self.D = self.D * m
            
        return baseClassifier,alphaCollect
    
    def predict(self, base_models, alphas, x_test):
        num = len(alphas)
        pred = np.zeros(len(x_test))
        for i in range(num):
            pred_temp = base_models[i].predict(x_test)
            pred_temp = pred_temp * 2.0 - 1.0

            pred = pred + pred_temp * alphas[i]
        for i in range(len(pred)):
            if (pred[i] > 0.0):
                pred[i] = 1
            else:
                pred[i] = 0
        return pred
            

    def tenFoldCV(self):
        base_list = [1, 5, 10, 100]
        kf = KFold(n_splits=10)
        for base_num in base_list:
            print('base_num == ', base_num)
            fold = 0
            for train_index, test_index in kf.split(self.X):
                fold += 1
                x_train = []
                y_train = []
                x_test = []
                y_test = []
                for item in train_index:
                    x_train.append(self.X[item])
                    y_train.append(self.y[item])
                for item in test_index:
                    x_test.append(self.X[item])
                    y_test.append(self.y[item])
                
                x_train = np.array(x_train)
                y_train = np.array(y_train)
                x_test = np.array(x_test)
                y_test = np.array(y_test)
                base_models, alphas = self.fit(x_train, y_train, base_num)
                test_index = test_index+1
                pred = self.predict(base_models, alphas, x_test)
                
                np.savetxt('experiments/base%d_fold%d.csv' % (base_num, fold), np.c_[test_index,pred], fmt = '%d,%d', delimiter=",")

ada = AdaBoost(X,y)
ada.tenFoldCV()