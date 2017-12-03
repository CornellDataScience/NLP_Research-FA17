import tensorflow as tf
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('../'))
from dl_style_transfer.layers.layers import xavier_initializer


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, num_reconstructions, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.num_reconstructions = num_reconstructions
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda

        # Placeholders for input, output and dropout
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.variable_scope("embedding"):

            self.reconstructions = tf.get_variable(
                name="Reconstructions",
                initializer=xavier_initializer([num_reconstructions, sequence_length, embedding_size]))
            self.embedded_chars_expanded = tf.expand_dims(self.reconstructions, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        self.activations = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=True):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.get_variable(name="W", shape=filter_shape, trainable=False)
                b = tf.get_variable(name="b", shape=[num_filters], trainable=False)
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                self.activations.append(h)
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, axis=3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.variable_scope("output", reuse=True):
            W = tf.get_variable(
                "W", shape=[num_filters_total, num_classes], trainable=False)
            b = tf.get_variable(name="b", shape=[num_classes], trainable=False)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        self.init_op = tf.variables_initializer([self.reconstructions])

