import tensorflow as tf

def rbf_kernel(x, h):
    dists = squared_distance(x)
    return tf.exp(-dists / h)

# it's a pun!
def squared_distance(x):
    dists = tf.reduce_sum(tf.square(x), axis=1, keep_dims=True)
    return dists - 2*tf.matmul(x, tf.transpose(x)) + tf.transpose(dists)
