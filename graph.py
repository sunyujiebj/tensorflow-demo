import tensorflow as tf
c = tf.constant([[2,3], [4,5]], name="const1", dtype=tf.int64)


# Launch the graph in a session.
sess = tf.Session()
if c.graph is tf.get_default_graph():
    print("The graph of c is the default graph of the context")
# print(sess.run(c))

