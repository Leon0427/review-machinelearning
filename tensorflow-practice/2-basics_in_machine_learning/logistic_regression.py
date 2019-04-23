#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/23 16:10
# @Author  : liangxiao
# @Site    : 
# @File    : logistic_regression.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tempfile
import urllib
import pandas as pd
import os
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist


######################################
######### Necessary Flags ############
######################################
tf.app.flags.DEFINE_string(
    "train_path",os.path.dirname(os.path.abspath(__file__)) + "/train_logs",
    "directory where event logs are written to."
)
tf.app.flags.DEFINE_string(
    "checkpoint_path",os.path.dirname(os.path.abspath(__file__)) + "/checkpoints",
    "directory where checkpoints are written to."
)

tf.app.flags.DEFINE_integer(
    "max_num_checkpoint",10,
    "Maximum number of checkpoints that TensorFlow will keep."
)

tf.app.flags.DEFINE_integer("num_classes", 2, "Number of model clones to deploy.")

tf.app.flags.DEFINE_integer("batch_size", int(np.power(2, 9)), "batch size.")

tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs for training.")

##########################################
######## Learning rate flags #############
##########################################
tf.app.flags.DEFINE_float("initial_learning_rate", 0.001, "initial learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "learning rate decay factor.")
tf.app.flags.DEFINE_float("num_epochs_per_decay", 1, "Number of epoch pass to decay learning rate.")

#########################################
########## status flags #################
#########################################
tf.app.flags.DEFINE_boolean("is_training", False, "Training/Testing.")
tf.app.flags.DEFINE_boolean("fine_tuning", False, "Fine tuning is desired or not?")
tf.app.flags.DEFINE_boolean("online_test", True, "Online testing?")
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Automatically put the variables on CPU if there's no "
                                                           "GPU support.")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Demonstrate which variables are on what device.")

# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS

################################################
################# handling errors!##############
################################################
if not os.path.abspath(FLAGS.train_path):
    raise ValueError('You must assign absolute path for --train_path')

if not os.path.abspath(FLAGS.checkpoint_path):
    raise ValueError('You must assign absolute path for --checkpoint_path')

# Download and get MNIST dataset(available in tensorflow.contrib.learn.python.learn.datasets.mnist)
# It checks and download MNIST if it's not already downloaded then extract it.
# The 'reshape' is True by default to extract feature vectors but we set it to false to we get the original images.

########################
### Data Processing ####
########################
# Organize the data and feed it to associated dictionaries.
data={}
(train_image,train_label),(test_image,test_label) = mnist.load_data()
train_image.resize([train_image.shape[0],train_image.shape[1]*train_image.shape[2]])
test_image.resize([test_image.shape[0],test_image.shape[1]*test_image.shape[2]])
data['train/image'],data['train/label'],data['test/image'],data['test/label'] = train_image, train_label, test_image, test_label

# only extract 1 or 0
def extract_samples_Fn(data):
    index_list = []
    for sample_index in range(data.shape[0]):
        label = data[sample_index]
        if label == 1 or label == 0:
            index_list.append(sample_index)
    return index_list


index_list_train = extract_samples_Fn(data["train/label"])
index_list_test = extract_samples_Fn(data["test/label"])

data['train/image'] = data['train/image'][index_list_train]
data['train/label'] = data['train/label'][index_list_train]
data['test/image'] = data['test/image'][index_list_test]
data['test/label'] = data['test/label'][index_list_test]

dimension_of_train = data['train/image'].shape
# print(demension_of_train)

num_train_samples = dimension_of_train[0]
num_features = dimension_of_train[1]

#######################################
########## Defining Graph ############
#######################################
graph = tf.Graph()
with graph.as_default():
    ###################################
    ########### Parameters ############
    ###################################

    global_step = tf.Variable(0,name="global_step", trainable=False)

    decay_steps = int(num_train_samples / FLAGS.batch_size * FLAGS.num_epochs_per_decay)

    learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               FLAGS.learning_rate_decay_factor,
                                               staircase=True,
                                               name="exponential_decay_learning_rate")

    ###############################################
    ########### Defining place holders ############
    ###############################################
    image_place = tf.placeholder(tf.float32, shape= ([None, num_features]),name= 'image')
    label_place = tf.placeholder(tf.int32, shape = ([None,]),name='gt')
    print(label_place)
    label_one_hot = tf.one_hot(label_place, depth=FLAGS.num_classes, axis=-1)
    print(label_one_hot)
    dropout_param = tf.placeholder(tf.float32)

    ##################################################
    ########### Model + Loss + Accuracy ##############
    ##################################################
    logits = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs=FLAGS.num_classes, scope='fc')
    print(logits)

    # Define loss
    with tf.name_scope("loss"):
        loss_tensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label_one_hot))

    # Accuracy
    # Evaluate the model
    prediction_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))

    # Accuracy calculation
    accuracy = tf.reduce_mean(tf.cast(prediction_correct, tf.float32))