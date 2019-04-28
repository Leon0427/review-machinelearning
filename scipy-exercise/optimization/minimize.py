#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/28 11:40
# @Author  : liangxiao
# @Site    : 
# @File    : minimize.py
# @Software: PyCharm

import numpy as np
from scipy.optimize import minimize

def rosen(x):
    """The RosenBrock function."""
    return sum(100.0* (x[1:]-x[:-1])**2.0 + (1-x[:-1])**2.0)

# initial res
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

# use different solver to minimize ()
res = minimize(rosen,x0,method="nelder-mead",options={'xtol': 1e-8, 'disp': True})
res = minimize(rosen,x0,method="Powell",options={'xtol': 1e-8, 'disp': True})

###########################################
### use derivatives to speed solving up ###
###########################################
def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm - xm_m1**2) - 400*xm*(xm_p1 - xm**2) - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der

# Broyden-Fletcher-Goldfarb-Shanno algorithm (method='BFGS')
res = minimize(rosen, x0, method="BFGS", jac=rosen_der, options={'gtol':1e-8,'disp':True,'maxiter':10})

# Newton-Conjugate-Gradient algorithm (method='Newton-CG')
