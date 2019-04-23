#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/22 21:54
# @Author  : liangxiao
# @Site    : 
# @File    : 1_basic_math.py
# @Software: PyCharm
import tensorflow as tf
import os
import numpy as np

a = tf.constant(5.0, name="a")
b = tf.constant(10.0, name="b")

x = tf.add(a, b, name="add")
y = tf.div(a, b, name="divide")

c = tf.constant(np.array([[1.,3.,5.],[3,6,8]]), name="c")

with tf.Session() as sess:
    print("a=",sess.run(a))
    print("b=",sess.run(b))
    print("a+b=", sess.run(x))
    print("a/b=", sess.run(y))
    print("c=", sess.run(c))
    print("max(c)=", sess.run(tf.argmax(c,0)))
    print("int(c)=", sess.run(tf.cast(c, tf.int32)))

sess.close()
