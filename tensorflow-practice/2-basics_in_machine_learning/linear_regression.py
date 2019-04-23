#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/23 11:32
# @Author  : liangxiao
# @Site    : 
# @File    : linear_regression.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd
import os
from sklearn.utils import check_random_state

n = 50
XX = np.arange(n)
rs = check_random_state(0)
YY = rs.randint(-10,10,size=(n,)) + 2.0 * XX
# axis of np.stack is like insert a axis to some place
# if XX.shape = (3,3) and YY.shape = (3,3)
# np.stack([XX,YY],axis=0).shape = (2,3,3)
# np.stack([XX,YY],axis=1).shape = (3,2,3)
# np.stack([XX,YY],axis=2).shape = (3,3,2)
data = np.stack([XX,YY],axis=1)

n_epochs = 50

# parameter in variable
W = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")


# data in placeholder
def inputs():
    """
    define the place_holders.
    :return return the data and label place holders
    """
    X = tf.placeholder(tf.float32, name = "X")
    Y = tf.placeholder(tf.float32, name = "Y")
    return X, Y


# model defined by formula
def inference(X):
    """
    Forward passing the X.
    :param X: Input.
    :return: X * W + b
    """
    return X*W + b


# loss is diff between pred and label
def loss(X,Y):
    """
    comparing the loss by comparing the predicted value to the actual label.
    :param X: the input.
    :param Y: the labels.
    :return: the loss over the samples.
    """
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y,Y_predicted))/(2*data.shape[0])


# train is the process to minimize loss, and loss is a function
def train(loss):
    learning_rate = 0.0001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 1. get placeholder for input
    X,Y = inputs()

    # 2. train loss is calculated by loss function
    train_loss = loss(X,Y)

    # train process is to optimize train loss
    train_op = train(train_loss)

    # iter n_epochs to train the model(modify parameters) to get a better pred
    for epoch_num in range(n_epochs):
        loss_value, _ = sess.run([train_loss, train_op], feed_dict={X:data[:,0],Y:data[:,1]})
        print("epoch %d, loss=%f"%(epoch_num+1, loss_value))
        # save the parameter
        wcoeff, bias =  sess.run([W,b])

input_values = data[:,0]
labels = data[:,1]
print(input_values, wcoeff,bias)
pred = input_values * wcoeff + bias
print(input_values.shape)

plt.plot(input_values, pred,label="predicted")
plt.plot(input_values, labels, "ro", label="main")
plt.legend()
plt.show()



