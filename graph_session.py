import tensorflow as tf
const1 = tf.constant([[2,2]])
const2 = tf.constant([[4],[4]])

multiple = tf.matmul(const1, const2)

sess = tf.Session()
result = sess.run(multiple)
print(result)

if const1.graph is tf.get_default_graph():
    print("const1 is default graph")
sess.close()

with tf.Session() as sess:
    result2 = sess.run(multiple)
    print("multiple %s " % result2)


