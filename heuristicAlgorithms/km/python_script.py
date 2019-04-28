#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/28 14:42
# @Author  : liangxiao
# @Site    : 
# @File    : python_script.py
# @Software: PyCharm

from scipy.optimize import linear_sum_assignment
import numpy as np
import sys
import json

if __name__ == '__main__':
    arr_file = sys.argv[1]
    arr_str = open(arr_file,"r").readlines()
    arr = json.loads(arr_str)
    val = np.array(arr)
    cost = 1 - val
    res = linear_sum_assignment(cost)
    print(res)
    # print(json.dumps([[1,2],[3,4]]))