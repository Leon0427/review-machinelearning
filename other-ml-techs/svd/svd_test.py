#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/4 11:01
# @Author  : liangxiao
# @Site    : 
# @File    : svd_test.py
# @Software: PyCharm

import numpy as np
from numpy.linalg import svd

"""all in all, svd's equation clearly illustrated usage of itself:
A = UΣV
A∈R(m*n)
U∈R(m*m)
Σ∈R(m*n)
V∈R(n*n)
values at the diagonal of Σ is called singular values,
basically they are in a descending order
we could chose first k strongest singular values to approximate A
A = U'Σ'V' 
A∈R(m*n)
U∈R(m*k)
Σ∈R(k*k)
V∈R(k*n)
"""
if __name__ == '__main__':
    data_shape = (7,4)
    data = np.random.randint(1,10,data_shape)
    print "the original data: \n%s"%data
    u,sigma,v = svd(data)
    print "singular values of data: \n%s"%sigma
    sigma_mat = np.zeros(data_shape)
    # only use first 2 singular values
    sigma_mat[0,0] = sigma[0]
    sigma_mat[1,1] = sigma[1]
    sigma_mat[2,2] = sigma[2]
    _data = u.dot(sigma_mat).dot(v)
    print "data correspond to first 3 singular values: \n%s "%_data
    # use all singular values
    sigma_mat[3,3] = sigma[3]
    print "data correspond to all sigular values: \n%s" % u.dot(sigma_mat).dot(v)