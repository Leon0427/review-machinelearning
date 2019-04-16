#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 10:17
# @Author  : liangxiao
# @Site    : 
# @File    : pca.test.py
# @Software: PyCharm

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris_data = load_iris()
X = iris_data.data
y = iris_data.target
pca = PCA(n_components=3)
X_ = pca.fit_transform(X)
lr1 = LogisticRegression()
lr2 = LogisticRegression()
lr1.fit(X,y)
lr2.fit(X_,y)
print y
print lr1.predict(X)
print lr2.predict(X_)
