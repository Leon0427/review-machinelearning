#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 19:57
# @Author  : liangxiao
# @Site    : 
# @File    : HowToCustomizeCallbacks.py
# @Software: PyCharm

import xgboost as xgb
dtrain = xgb.DMatrix()
def my_callback():
    def callback(env):
        model = env.model
        pass
    callback.before_iteration = True
    return callback

bst = xgb.train({},dtrain,callbacks=[my_callback()])