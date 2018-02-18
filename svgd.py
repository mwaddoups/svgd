import tensorflow as tf
import numpy as np
from kernels import rbf_kernel

class SVGD: 
    def __init__(self, initial_particles, target_pdf):
        self.target_pdf = target_pdf
        self.particles = initial_particles
        
    def rbf_kernel(self):
        dists = self.pairwise_distance(self.particles)
        h = tf.stop_gradient(self.get_h(dists))
        return tf.exp(-dists / h)

    def get_h(self, dists):
        v = tf.reshape(dists, [-1])
        m = v.get_shape()[0]//2
        median = tf.nn.top_k(v, m).values[m-1]

        return median / tf.log(tf.cast(dists.get_shape()[0], tf.float32))

    def pairwise_distance(self, x):
        dists = tf.reduce_sum(tf.square(x), axis=1, keep_dims=True)
        return dists - 2*tf.matmul(x, tf.transpose(x)) + tf.transpose(dists)

    def get_gradients(self):
        N = tf.cast(self.particles.get_shape()[0], tf.float32)
        xgrads = tf.gradients(self.target_pdf(self.particles), self.particles)[0]

        kernel = self.rbf_kernel()
        # trick to get kernel grads - stationary only!
        # added minus here - work out why!!!
        kgrads = -0.5 * (tf.gradients(kernel, self.particles)[0] + tf.gradients(tf.diag_part(kernel), self.particles)[0])

        step = (tf.matmul(kernel, xgrads) + kgrads) / N

        return step
    
    def compile(self, optimizer=None):
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(0.01)
            
        step = self.get_gradients()
        train_op = optimizer.apply_gradients([(-step, self.particles)])

        self.train_op = train_op
    
    def run(self, n_iters=1000, call_every=None):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(n_iters):
                sess.run(self.train_op)

                if call_every is not None:
                    func, ns = call_every
                    if (i + 1) % ns == 0:
                        func(sess.run(self.particles), sess)
                 
            return sess.run(self.particles)
