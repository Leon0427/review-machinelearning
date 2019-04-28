#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/24 10:26
# @Author  : liangxiao
# @Site    : 
# @File    : basics.py
# @Software: PyCharm

import numpy as np

# Create a array
a = np.random.randint(1,9,(4,2,5))
b = np.full((4,2,5),3)
c = np.eye(3,2)

# get max/min
min_of_a_0 = a.mean(axis=0) # output shape = (2,5)
min_of_a_1 = a.mean(axis=1) # output shape = (4,5)
min_of_a_2 = a.mean(axis=2) # output shape = (4,2)
min_of_a = a.mean() # output is single number
print(min_of_a)
print(min_of_a_0)
print(min_of_a_1)
print(min_of_a_2)

# get sum
sum_of_a_0 = a.sum(axis=0)
sum_of_a_1 = a.sum(axis=1)
sum_of_a_2 = a.sum(axis=2)
print(sum_of_a_0)
print(sum_of_a_1)
print(sum_of_a_2)

# diag
print("-------- diag ---------")
diag_1 = np.diag(range(1,4),1)
diag_2 = np.diag(range(1,4),0)
diag_3 = np.diag(range(1,4),2)
print(diag_1)
print(diag_2)
print(diag_3)