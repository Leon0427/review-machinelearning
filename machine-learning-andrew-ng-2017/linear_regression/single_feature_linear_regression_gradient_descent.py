#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/1 22:50
# @Author  : Leon
# @Site    : 
# @File    : single_feature_linear_regression_gradient_descent.py

import numpy as np
import matplotlib.pyplot as plt

# 1. generate data based on distribution: y = θ_1 * x + θ_0 + noise
X = np.linspace(0, 10, 400)
Y = 0.345 + 3.235 * X  + np.random.normal(0, 3, 400)

# 2. init weights and learning-rate
w = np.zeros(2)
a = 0.0001

J = 1 / (2*400) * ( w[0]*X + w[1] - Y )*( w[0]*X + w[1] - Y )

# 3.stop iter when J < 1 or iter > 100000 or loss decrease < 0.0001
iter = 0
while(sum(J)>1):
    tmp_w0 = w[0]
    tmp_w1 = w[1]
    w[0] = w[0] - sum(a*(1/400.0 * (tmp_w0*X + tmp_w1 - Y) * X ))
    w[1] = w[1] - sum(a*(1/400 * (tmp_w0*X + tmp_w1 - Y) * 1))
    newJ = 1 / (2 * 400) * (w[0] * X + w[1] - Y) * (w[0] * X + w[1] - Y)
    if np.abs(sum(newJ) - sum(J)) < 0.0001 or iter>100000:
        print(sum(newJ),sum(J))
        print(iter)
        break
    J = newJ
    iter += 1
    if iter % 1000 == 0:
        print("iter:{0}, loss:{1}".format(iter, sum(J)))

# 4. print w and J
print(sum(J))
print (w)



