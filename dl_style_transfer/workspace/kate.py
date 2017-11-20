import tensorflow as tf
import numpy as np
import math


def zero_if_false(condition, vals):
    return tf.select(condition, vals, tf.zeros(tf.shape(vals)))


def hunger_games(tensors, k, alpha):
    """Takes a layer of a neural network and applies the tanh activation function; it
    then applies KATE style competition and returns the result"""
    with tf.variable_scope("KATE Competition"):
        tan = tf.tanh(tensors, name="K-Activated")
        pos = zero_if_false(tf.greater(tan, 0), tan)
        neg = zero_if_false(tf.less(tan, 0), tan)
        pos_adder = tf.reduce_sum(pos)
        neg_adder = tf.reduce_sum(neg)
        pos_val, _ = tf.nn.top_k(pos, math.ciel(k / 2)) + alpha * pos_adder
        neg_val, _ = -tf.nn.top_k(-neg, math.floor(k / 2)) + alpha * neg_adder
    return tf.concat([pos_val, neg_val], 0)


class Kate:

    def __init__(self, vocab_size, k, alpha, hsize=10):
        self.bag = tf.placeholder(tf.placeholder(tf.float32, [1, vocab_size]), 'Bag')
        W = tf.Variable(tf.truncated_normal(0, stddev=.1, shape=[vocab_size, hsize]))
        self.layer = tf.matmul(self.bag, W)
        self.k_layer = hunger_games(self.layer, k, alpha)
        out_w = tf.Variable(tf.truncated_normal(0, stddev=.1, shape=[k, vocab_size]))
        self.out = tf.matmul(self.k_layer, out_w)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss(self.bag, self.out))

    def loss(self, x, x_hat):
        return tf.reduce_sum(x * tf.log(x_hat) + (1 - x) * tf.log(1 - x_hat))

    def start(self):
        self.sess = tf.InteractiveSession()

    def reset(self):
        self.shutdown()
        self.start()

    def shutdown(self):
        self.sess.close()

    def lognorm(self, x):
        log = np.log(1 + x)
        return log / np.max(log)

    def fit(self, x):
        pass

    def predict(self, x):
        pass

if __name__ == '__main__':
    k = Kate(10000, 10, 6.6, 10000)
    