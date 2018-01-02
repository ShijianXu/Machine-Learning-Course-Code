# -*- coding: utf-8 -*-
"""
Created on Sat May 13 23:21:18 2017

@author: Xu Shijian
"""
import numpy as np

def predict(Xt,model):
    pred = []
    
    dual_coef = model.dual_coef_[0]
    rho = model.intercept_[0]
    gamma = model.gamma
    vectors = (model.support_vectors_)
    #print(len(vectors))    
    
    n = len(vectors)

    for j in range(len(Xt)):
        x = np.array(Xt[j])
        ans = 0
        for i in range(n):
            xi = np.array(vectors[i])
            x_t = xi-x
            norm = np.inner(x_t, x_t)
            ans+=dual_coef[i]*(np.exp(-gamma*(norm)))
        ans+= rho
        if ans < 0:
            pred.append(0)
        else:
            pred.append(1)
            
    return pred