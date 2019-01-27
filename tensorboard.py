# -*- coding: UTF-8 -*-
import tensorflow as tf
W = tf.Variable(2.0, dtype=tf.float32, name="Weight")
b = tf.Variable(1.0, dtype=tf.float32, name="Bias")
x = tf.placeholder(dtype=tf.float32, name='Input')

with tf.name_scope("Output"):
    y = W * x + b

const = tf.constant(2.0)

path = "./log"

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(path, sess.graph)
    result = sess.run(y, {x:3.0})
    print("y = %s" % result)


#tensorboard --logdir=log