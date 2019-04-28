#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/28 10:35
# @Author  : liangxiao
# @Site    : 
# @File    : compare_munkres_with_sklearn.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from munkres import Munkres
from scipy.optimize import linear_sum_assignment
import time

# 经验证， linear_sum_assignment 比 mankres 快一点，结果一样

def get_val_and_cost(shape=(300,200)):
    val = np.abs(np.random.normal(0,0.1,size=shape))
    val[val>1.] = 1.
    cost = 1 - val
    return val,cost

val,cost = get_val_and_cost((300,300))
time1 = time.time()
print(time1)
ans1 = linear_sum_assignment(cost)
time2 = time.time()
km = Munkres()
# ans2 = km.compute(cost)
time3 = time.time()

print("lsa:%s" %(time2-time1))
print("km:%s" %(time3-time2))



print(ans1[0])
print(np.sort(ans1[1]))
# print(ans2)
