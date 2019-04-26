#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/26 16:19
# @Author  : liangxiao
# @Site    : 
# @File    : isotonic_regression.py
# @Software: PyCharm

from sklearn.isotonic import IsotonicRegression
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,100,1)
print(x)
y = 2 * x + (0.5-np.random.normal(0,1.5,x.shape))*10

ir = IsotonicRegression()
y_ = ir.fit_transform(x,y)
print(x)
print(y)
plt.scatter(x,y)
plt.plot(x,y_,color="red")
plt.show()