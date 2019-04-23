#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/23 10:08
# @Author  : liangxiao
# @Site    : 
# @File    : 1_1_variables.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.python.framework import ops

#######################################
######## Defining Variables ###########
#######################################

# Create three variables with some default values.
weights = tf.Variable(tf.random_normal([2,3],stddev=0.1,name="weights"))
biases = tf.Variable(tf.zeros([3]), name="biases")
custom_variable = tf.Variable(tf.zeros([3]), name="custom")
# Get all the variables' tensors and store them in a list.
all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

############################################
######## Customized initializer ############
############################################
## Initialation of some custom variables.
## In this part we choose some variables and only initialize them rather than initializing all variables.
variable_list_custom = [weights, custom_variable]
init_custom_op = tf.variables_initializer(var_list=variable_list_custom)

########################################
######## Global initializer ############
########################################
init_all_op = tf.global_variables_initializer()
init_all_op_2 = tf.variables_initializer(var_list=all_variables_list)

##########################################################
######## Initialization using other variables ############
##########################################################
weightsNew = tf.Variable(weights.initialized_value(), name="weightsNew")
init_weightsNew_op = tf.variables_initializer(var_list=[weightsNew])

with tf.Session() as sess:
    sess.run(init_all_op)
    sess.run(init_custom_op)
    sess.run(init_weightsNew_op)