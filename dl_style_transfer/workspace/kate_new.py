from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os
import math

from dl_style_transfer.layers.layers import *
from time import time, strftime


class Kate:

    def __init__(self, embedding_size_in, embedding_size_out, k, alpha, learning_rate=0.001, load_model=None):
        """Initializes the KATE model. Does not perform any training.

        Args:
            embedding_size_in:  The size of the input embeddings. For word embeddings, this is the vocabulary size.
            embedding_size_out: The size of the output embeddings. This is the size of the encoded vectors.
            k:                  The number of neurons to keep in the k-competitive layer.
            alpha:              Sets the intensity of the energy redistribution in the k-comptitive layer.
            learning_rate:      The learning rate for training.
            load_model:         A string giving the path to the model to load. If `None`, the model's weights are
                                randomly initialized to start training from scratch.
        """
        self._embedding_size_in = embedding_size_in
        self._embedding_size_out = embedding_size_out
        self._k = k
        self._alpha = alpha

        print("Constructing KATE architecture...")
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._x = tf.placeholder(tf.int32, shape=[None], name='X')
            self._phase_train = tf.placeholder(tf.bool, name='Phase')

            self._embedding = tf.one_hot(self._x, depth=embedding_size_in, name='Embedding')  # `[batch, embedding_size_in]`

            with tf.variable_scope('Encoder'):
                fc1, var_dict = fc(self._embedding, embedding_size_out, activation=tf.nn.tanh, scope='FC1')

                self._encoded = k_comp(fc1, self._phase_train, k, alpha=alpha, scope='Encoded')

            with tf.variable_scope('Decoder'):
                weights = tf.transpose(var_dict['Weights'])  # `[embedding_size_out, embedding_size_in]`
                bias = tf.get_variable('Scores', initializer=xavier_initializer((embedding_size_in,)))
                scores = tf.matmul(self._encoded, weights) + bias  # `[batch*max_seq_len, embedding_size_in]

                self._decoded = tf.nn.softmax(scores, name='Decoded')
                self._argmax = tf.argmax(self._decoded, axis=-1, name='Decoded-Argmax')

            with tf.variable_scope('Loss'):
                self._loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=self._x, name='Loss'))
                self._train_step = tf.train.AdamOptimizer(learning_rate).minimize(self._loss)

            self._sess = tf.Session()
            with self._sess.as_default():
                self._saver = tf.train.Saver()
                if load_model is not None:
                    print("Restoring Model...")
                    load_model = os.path.abspath(load_model)
                    self._saver.restore(self._sess, load_model)
                    print("Model Restored!")
                else:
                    print("Initializing model...")
                    self._sess.run(tf.global_variables_initializer())
                    print("Model Initialized!")

    def train(self, x_train, n_epochs, batch_size, start_stop_info=True, progress_interval=5):
        """Trains the model on the batch of data provided. Typically called before inference.

        Args:
            x_train:           A numpy ndarray that contains the data to train over. If shaped  `[batch_size]`, the data
                               is interpreted as a list of indices, and is internally expanded into one-hot vectors of
                               length `embedding_size_in`. If shaped `[batch_size, embedding_size_in]`, then the data is
                               interpreted as a batch of embedding vectors of length `embedding_size_in`.
            n_epochs:          The number of full passes over the provided dataset to perform until training is
                               considered to be complete.
            batch_size:        The size of the batch to use when training. Larger sizes mean a more stable loss function
                               which might enable a larger value for the learning rate. However, the larger the batch,
                               the more memory will be used and the slower the training speed will be per iteration.
            start_stop_info:   If true, print when the training begins and ends.
            progress_interval: If not `None`, then this is the minimum interval of time between printing the progress
                               of the model.
        Returns:
            The loss value after training.
        """
        training_size = x_train.shape[0]

        key = [self._x if len(x_train.shape) is 1 else self._embedding]  # Choose which tensor to feed to based on shape

        # Training loop for parameter tuning
        if start_stop_info:
            print("Starting training for %d epochs" % n_epochs)

        batch_per_epoch = math.ceil(training_size/batch_size)
        n_batch = n_epochs*batch_per_epoch
        last_time = time()
        for epoch in range(n_epochs):
            perm = np.random.permutation(training_size)
            for i in range(0, training_size, batch_size):
                idx = perm[i:i + batch_size]
                x_batch = x_train[idx]
                _, loss_val = self._sess.run([self._train_step, self._loss], feed_dict={key: x_batch, self._phase_train: False})
                current_time = time()
                if progress_interval is not None and (current_time - last_time) >= progress_interval:
                    last_time = current_time
                    print("Current Loss Value: %.10f, Percent Complete: %.4f" %
                          (loss_val, (epoch*batch_per_epoch + i) / n_batch * 100))
        if start_stop_info:
            print("Completed Training.")
        return loss_val

    def encode(self, raw_data):
        """Encodes the batch of data provided. Typically called after the model is trained.

        Args:
            raw_data: A numpy ndarray of the data to encode. Should match the data format of `train()`.

        Returns:
            A numpy ndarray of the data, with shape `[batch_size, embedding_size]`
        """
        with self._sess.as_default():
            return self._sess.run(self._encoded, feed_dict={self._x: raw_data, self._phase_train: False})

    def decode(self, encodings, do_argmax=True):
        """Decodes the batch of data provided. Typically called after the model is trained.

        Args:
            encodings: A numpy ndarray of the data to decode. Should have shape `[batch_size, embedding_size]`.
            do_argmax: Whether or not to argmax the vectors returned to predict the word.

        Returns:
            A numpy ndarray of the decoded data. If `do_argmax` is `False`, then the data will have shape
            `[batch_size, embedding_size_in]`. If `do_argmax` is `True` the shape is `[batch_size]`.
        """
        with self._sess.as_default():
            return self._sess.run(
                self._argmax if do_argmax else self._decoded,
                feed_dict={self._encoded: encodings, self._phase_train: False})

    def reconstruct(self, raw_data, do_argmax=True):
        """Reconstructs the batch of data provided by encoding and then decoding it. Typically called after the model
        is trained.

        Args:
            raw_data:  A numpy ndarray of the data to reconstruct. Should match the data format of `train()`.
            do_argmax: Whether or not to argmax the vectors returned to reconstruct the word.

        Returns:
            A numpy ndarray of the reconstructed data. If `do_argmax` is `False`, then the data will have shape
            `[batch_size, embedding_size_in]`. If `do_argmax` is `True` the shape is `[batch_size]`.
        """
        with self._sess.as_default():
            return self._sess.run(
                self._argmax if do_argmax else self._decoded,
                feed_dict={self._x: raw_data, self._phase_train: False})

    def save_model(self, save_path=None):
        """Saves the model in the specified file.

        Args:
            save_path: The relative path to the file. By default, it is
                       saved/KATE-Year-Month-Date_Hour-Minute-Second.ckpt
        """
        with self._sess.as_default():
            print("Saving Model")
            if save_path is None:
                save_path = "saved/KATE-%s.ckpt" % strftime("%Y-%m-%d_%H-%M-%S")
            dirname = os.path.dirname(save_path)
            if dirname is not '':
                os.makedirs(dirname, exist_ok=True)
            save_path = os.path.abspath(save_path)
            path = self._saver.save(self._sess, save_path)
            print("Model successfully saved in file: %s" % path)
