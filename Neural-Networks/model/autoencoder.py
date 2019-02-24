#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__  =   "Jifu Zhao"
__email__   =   "jzhao59@illinois.edu"
__date__    =   "09/05/2017"
__modify__  =   "09/05/2017"
"""

import numpy as np
import tensorflow as tf


class Autoencoder(object):
    """
        Self-defined class for Autoencoders using tensorflow
        Currently it only support 5-layer structure
    """
    def __init__(self, layers=(1024, 512, 256, 512, 1024),
                 run_id='Autoencoder', seed=None):
        """ initialize Autoencoder """
        if seed is not None:
            tf.set_random_seed(seed)
            np.random.seed(seed)

        # re-set the graph
        tf.reset_default_graph()

        self.layers = layers
        self.run_id = run_id
        self.x = None
        self.session = None
        self.encoded = None
        self.decoded = None
        self.loss = None
        self.train_loss = []
        self.cv_loss = []

        self._initialize()
        self.session = tf.Session()
        self.writer = tf.summary.FileWriter('./model/' + run_id,
                                            self.session.graph)

    def _reset(self, layers=(1024, 512, 256, 512, 1024), run_id='Autoencoder',
               seed=None):
        """ reset Autoencoder """
        self.__init__(layers, run_id, seed)

    def _initialize(self):
        """ initialize Autoencoder """
        with tf.name_scope('Data'):
            self.x = tf.placeholder(tf.float32, [None, self.layers[0]],
                                    name='X')

        with tf.name_scope('Encoder'):
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1)

            en_w1 = tf.get_variable('en_w1', [self.layers[0], self.layers[1]],
                                    initializer=initializer)
            en_b1 = tf.get_variable('en_b1', [self.layers[1]],
                                    initializer=initializer)

            en_w2 = tf.get_variable('en_w2', [self.layers[1], self.layers[2]],
                                    initializer=initializer)
            en_b2 = tf.get_variable('en_b2', [self.layers[2]],
                                    initializer=initializer)

            # build the encoder
            encoder_tmp = tf.nn.relu(tf.matmul(self.x, en_w1) + en_b1)
            self.encoded = tf.nn.relu(tf.matmul(encoder_tmp, en_w2) + en_b2)

        with tf.name_scope('Decoder'):
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
            de_w1 = tf.get_variable('de_w1', [self.layers[2], self.layers[3]],
                                    initializer=initializer)
            de_b1 = tf.get_variable('de_b1', [self.layers[3]],
                                    initializer=initializer)

            de_w2 = tf.get_variable('de_w2', [self.layers[3], self.layers[4]],
                                    initializer=initializer)
            de_b2 = tf.get_variable('de_b2', [self.layers[4]],
                                    initializer=initializer)

            # build the decoder
            decoder_tmp = tf.nn.relu(tf.matmul(self.encoded, de_w1) + de_b1)
            self.decoded = tf.nn.relu(tf.matmul(decoder_tmp, de_w2) + de_b2)

        with tf.name_scope('Output'):
            # use the mean squared error
            diff = tf.squared_difference(self.x, self.decoded)
            self.loss = tf.reduce_mean(tf.reduce_sum(diff, axis=1))

    def train(self, train_x, learning_rate=0.001, steps=100, batch_size=1000,
              sub_steps=10, cv=None, reset=False, run_id='Autoencoder',
              layers=(1024, 512, 256, 512, 1024), seed=None):
        """ train Autoencoder """
        if reset is True:
            self._reset(layers, run_id, seed)

        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        init = tf.global_variables_initializer()

        self.session.run(init)
        for step in range(steps):
            batches = self._get_batch(batch_size, train_x)
            j = 0  # limit the sub-training process to be within sub_steps
            for batch_x in batches:
                j += 1
                self.session.run(self.opt, feed_dict={self.x: batch_x})
                if j == sub_steps:
                    # use only sub_steps batches to train the model
                    break

            # evaluate on training and cv set
            tmp_loss = self.session.run(self.loss, feed_dict={self.x: train_x})
            self.train_loss.append(tmp_loss)

            if cv is not None:
                tmp_loss = self.session.run(self.loss, feed_dict={self.x: cv})
                self.cv_loss.append(tmp_loss)

        self.writer.close()

        return self.train_loss, self.cv_loss

    def predict(self, test_x):
        """ return test_loss, test_encoded, test_decoded """
        result = self.session.run([self.encoded, self.decoded, self.loss],
                                  feed_dict={self.x: test_x})

        return result

    def _get_batch(self, batch_size, train_x):
        """ split into batches """
        if batch_size is None or batch_size == len(train_x):
            return [train_x]

        split = len(train_x) // batch_size
        idx = list(range(len(train_x)))
        np.random.shuffle(idx)

        batches = []
        for i in range(split):
            batches.append(train_x[(i * batch_size): (i + 1) * batch_size, :])

        return batches

    def close(self):
        """ To visualize the Graph
            tensorboard --logdir=Autoencoder_test/
        """
        self.session.close()
