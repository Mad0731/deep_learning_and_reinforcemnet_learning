import argparse
import os
import time
# import pandas as pd
# from tensorflow.keras import layers
# from tensorflow.keras import initializers
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
num_decay = 43200


def nn_weights(shape, stddev=.1):
    """ Weight initialization """
    weights = tf.random.normal(shape, stddev=stddev)
    print("Weight shape:", shape)
    return tf.Variable(weights)


def nn_bias(shape, stddev=.1):
    """ Bias initialization """
    biases = tf.random.normal([shape], stddev=stddev)
    print("Bias shape:", [shape])
    return tf.Variable(biases)


def weights_save(sess, weights, biases, output_file, weight_from_model, num_layers, project_prefix):
    for i in range(0, num_layers + 1):
        weight_i = sess.run(weights[i])
        np.savetxt(output_file + weight_from_model + project_prefix + "w_" + str(i) + ".txt", weight_i, delimiter=',')
        bias_i = sess.run(biases[i])
        np.savetxt(output_file + weight_from_model + project_prefix + "b_" + str(i) + ".txt", bias_i, delimiter=',')
        print("Bias: ", i, " : ", bias_i)
    return


def weights_to_nn(output_file, weight_name_file_to_nn, num_layers, project_prefix):
    weights = []
    biases = []
    for i in range(0, num_layers + 1):
        weight_i = np.loadtxt(output_file + weight_name_file_to_nn + project_prefix + "w_" + str(i) + ".txt",
                              delimiter=',')
        w_i = tf.Variable(weight_i, dtype=tf.float32)
        weights.append(w_i)
        bias_i = np.loadtxt(output_file + weight_name_file_to_nn + project_prefix + "b_" + str(i) + ".txt",
                            delimiter=',')
        b_i = tf.Variable(bias_i, dtype=tf.float32)
        biases.append(b_i)
    return weights, biases


def forward_propagation(x, weights, biases, num_layers, dropout=False):
    htemp = None
    for i in range(0, num_layers):
        if i == 0:
            htemp = tf.nn.relu(tf.math.add(tf.matmul(x, weights[i]), biases[i]))
        else:
            htemp = tf.nn.relu(tf.math.add(tf.matmul(htemp, weights[i]), biases[i]))
        print("Bias: ", i, " : ", biases[i])
    # drop_out = tf.nn.dropout(htemp,0.9)
    y_hat = tf.math.add(tf.matmul(htemp, weights[-1]), biases[-1])
    print("Last bias: ", biases[-1])
    return y_hat
