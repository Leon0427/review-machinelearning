#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/11 23:37
# @Author  : Leon
# @Site    : 
# @File    : data_generator.py

"""
m opps , and n accs
m > 3n

input: m * n matrix (score)

output: 3*n matrix

"""

import numpy as np

m = 16
n = 4
np.random.seed(15)
input = np.random.random((16,4))
print (input)

def km(input):
    pass

# basic heuristic by randomly choosing answer
def heuristic(input,iter=100):
    better_score = 0.0
    better_ans = None
    cnt = 0
    while cnt<iter:
        ans = np.random.choice(m,(n,3),replace=False)
        new_score = 0.0
        for acc_idx, row_content in enumerate(ans):
            for opp_idx in row_content:
                new_score += input[opp_idx, acc_idx]
        if new_score>=better_score:
            better_score = new_score
            better_ans = ans
        cnt += 1
    return better_score,better_ans


final_score,final_ans = heuristic(input,10000)

print("final score is {0}, final ans is {1}".format(final_score, final_ans))

