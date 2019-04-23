import tensorflow as tf
import os
welcome = tf.constant("hello world")
with tf.Session() as sess:
    print("output:", sess.run(welcome))
sess.close()