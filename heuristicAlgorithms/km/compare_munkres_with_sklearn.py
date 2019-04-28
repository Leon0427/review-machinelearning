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

# 经验证， linear_sum_assignment 比 mankres 快一点，结果一样

def get_val_and_cost(shape=(100,100)):
    val = np.abs(np.random.normal(0,0.1,size=shape))
    val[val>1.] = 1.
    cost = 1 - val
    return val,cost

val,cost = get_val_and_cost((100,100))
ans1 = linear_sum_assignment(cost)
km = Munkres()
ans2 = km.compute(cost)

print(ans1)
print(ans2)
