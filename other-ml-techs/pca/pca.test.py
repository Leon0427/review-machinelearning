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
pca2 = PCA(n_components=3)
pca3 = PCA(n_components=2)
pca4 = PCA(n_components=1)
X2 = pca2.fit_transform(X)
X3 = pca3.fit_transform(X)
X4 = pca4.fit_transform(X)
lr1 = LogisticRegression()
lr2 = LogisticRegression()
lr3 = LogisticRegression()
lr4 = LogisticRegression()
lr1.fit(X,y)
lr2.fit(X2,y)
lr3.fit(X3,y)
lr4.fit(X4,y)
print y
print lr1.predict(X)
print lr2.predict(X2)
print lr3.predict(X3)
print lr4.predict(X4)

print "as we can see, as the n_components goes down, the precession of the model we build with X_i goes down as well"
