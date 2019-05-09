#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/30 10:28
# @Author  : liangxiao
# @Site    : 
# @File    : plot_contour.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt

def plot_contour(x,y,f):
    X,Y = np.meshgrid(x,y)
    Z = f(X,Y)
    print (Z)
    plt.contourf(X,Y,Z,cmap=plt.cm.hot,alpha=0.6)
    C = plt.contour(X,Y,Z)
    plt.clabel(C, inline=True, fontsize=10)
    plt.show()

if __name__ == '__main__':
    plot_contour(np.linspace(-5,5,100),np.linspace(-5,3,100),lambda x,y:x**2+y**2)