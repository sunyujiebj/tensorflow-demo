import tensorflow as tf
import numpy as np
hello = tf.constant('Hello, World!')
sess = tf.Session()
print(sess.run(hello))



zeros = np.zeros((3,4))
print(zeros)


ones = np.ones((5,6))
print(ones)

ident = np.eye(4)
print(ident)


var = tf.Variable(3)
var1 = tf.Variable(3, dtype=tf.int64)
print(var)
print(var1)

var2 = tf.Variable([3,4])
print(var2)
var3 = tf.Variable([1,2],[3,4])
print(var3)

const = tf.constant(3)
print(const)
