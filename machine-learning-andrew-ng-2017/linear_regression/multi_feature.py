#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/5 9:41
# @Author  : Leon
# @Site    : 
# @File    : multi_feature.py

import numpy as np

# 1. init data
theta_ = np.array([[1],[2],[3],[4],[5]],dtype="float64")
X = np.random.random(size=(1000,4))
X = np.hstack((np.ones(shape=(1000,1)),X))
y = X.dot(theta_) + np.random.normal(0,.1,(1000,1))

# test data
print(y[:5])

# shape of data
print(X.shape)
print(y.shape)

theta = np.zeros((5, 1))

y_pred = X.dot(theta)
# J = Σ(1/(2*1000))(y_pred - y)^2
# J' = Σ(1/1000)(y_pred - y)y_pred'
oldJ = -1
J = 1/2000 * np.sum(np.square(y_pred - y))
a = 0.01

iter = 0
while(J>0.001):
    iter+= 1
    if iter%10000==0:
        print("iteration: {0}, loss: {1}".format(iter, J))
    delta = np.sum(0.001 * (1 / 1000) * (X.dot(theta) - y) * X,axis=0)
    # print("delta %s"%delta)
    for i,d in enumerate(delta):
        theta[i,0] -= d
    y_pred = X.dot(theta)
    J = 1 / 2000 * np.sum(np.square(y_pred - y))
    # theta_i = theta_i - a * (1/1000)*(X.dot(theta) - y)*X[:,i]
    # theta = theta - X.dot(theta)
    if iter>200000:
        break
print("fact: %s"%theta_)
print("ans: %s"%theta)
