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

    #############################################
    ########### training operation ##############
    #############################################

    # Define optimizer by its default values
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # 'train_op' is a operation that is run for gradient update on parameters.
    # Each execution of 'train_op' is a training step.
    # By passing 'global_step' to the optimizer, each time that the 'train_op' is run, Tensorflow
    # update the 'global_step' and increment it by one!

    # gradient update
    with tf.name_scope('train_op'):
        gradient_and_variables = optimizer.compute_gradients(loss_tensor)
        train_op = optimizer.apply_gradients(gradient_and_variables, global_step=global_step)

    ############################################
    ############ Run the Session ###############
    ############################################
    sess_config = tf.ConfigProto(
        allow_soft_placement=FLAGS.all_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(graph=graph, config=sess_config)

    with sess.as_default():

        # the saver op.
        saver = tf.train.Saver()

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # the prefix for checkpoint files
        checkpoint_prefix = 'model'

        # If fine-tuning flag is 'True', the model will be restored
        if FLAGS.fine_tuning:
            saver.restore(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
            print("Model restored for fine-tuning...")

        ###################################################################
        ########## Run the training and loop over the batches #############
        ###################################################################

        # go through the batches
        test_accuracy = 0
        for epoch in range(FLAGS.num_epochs):
            total_batch_training = int(data['train/image'].shape[0] / FLAGS.batch_size)

            # go through the batches
            for batch_num in range(total_batch_training):
                #################################################
                ########## Get the training batches #############
                #################################################
                start_idx = batch_num * FLAGS.batch_size
                end_idx = start_idx + FLAGS.batch_size

                # Fit training with batch data
                train_batch_data, train_batch_label = data['train/image'][start_idx:end_idx],data['train/label'][start_idx, end_idx]

                ########################################
                ########## Run the session #############
                ########################################

                # run optimization op(backprop) and calculate batch loss and accuracy
                # when the tensor tensors['global_step'] is evaluated, it'll be incremented by 1.
                batch_loss, _, training_step = sess.run(
                    [loss_tensor, train_op, global_step],
                    feed_dict={image_place:train_batch_data,
                                label_place:train_batch_label,
                                dropout_param:0.5})

                ########################################
                ########## Write summaries #############
                ########################################

            #################################################
            ########## Plot the progressive bar #############
            #################################################
            print("Epoch" + str(epoch + 1) + ", Train Loss= " + \
                  "{:.5f}".format(batch_loss))

        ###########################################################
        ############ Saving the model checkpoint ##################
        ###########################################################

        # # the model will be saved when then training is done

        # create the path for saving the checkpoints.
        if not os.path.exists(FLAGS.checkpoint_path):
            os.mkdir(FLAGS.checkpoint_path)

        # save the model
        save_path = saver.save(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
        print("Model saved in file: %s" % save_path)

        ############################################################################
        ########## Run the session for pur evaluation on the test data #############
        ############################################################################

        # The prefix for checkpoint files
        checkpoint_prefix = 'model'

        # Restoring the saved files
        saver.restore(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
        print("Model restored...")

        # Evaluation of the model
        test_accuracy = 100*sess.run(accuracy, feed_dict={
            image_place:data['test/image'],
            label_place:data['test/label'],
            dropout_param:1.
        })

        print("Final Test Accuracy is %% %.2f" % test_accuracy)

