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
def reading_data(data_file, percenttest=.2, random_state=42):
    x_file = data_file + 'geometry.csv'
    y_file = data_file + 'rcs.csv'
    x_data = np.genfromtxt(x_file, delimiter=',', skip_header=1, usecols=[0, 1, 2, 3, 4, 5])
    print(x_data.shape)
    y_data = np.genfromtxt(y_file, delimiter=',', skip_header=1)
    print(y_data.shape)
    # print("First few lines of x_data and y_data:")
    # for i in range(2):
        # print("x_data[{}]: {}".format(i, x_data[i]))
        # print("y_data[{}]: {}".format(i, y_data[i]))
    x_mean = x_data.mean(axis=0)
    x_std = x_data.std(axis=0)
    x_data = (x_data-x_data.mean(axis=0))/x_data.std(axis=0)
    y_mean = y_data.mean(axis=0)
    y_std = y_data.std(axis=0)
    y_data = (y_data - y_data.mean(axis=0)) / y_data.std(axis=0)
    # Normalize train_X and train_Y
    #std_x = (x_data - x_data.mean(axis=0)) / x_data.std(axis=0)
    #std_y = (y_data - y_data.mean(axis=0)) / y_data.std(axis=0)
    print("Train_x:", x_data.mean)
    #x_mean = x_data.mean(axis=0)
    #x_std = x_data.std(axis=0)
    train_x, x_test, train_y, y_test = train_test_split(x_data, y_data, test_size=percenttest, random_state=random_state)
    print("First few lines of train_x and train_y:")
    for i in range(2):
        print("train_x[{}]: {}".format(i, train_x[i]))
        print("train_y[{}]: {}".format(i, train_y[i]))
    # for i in range(5):
        # print("test_x[{}]: {}".format(i, x_test[i]))
        # print("test_y[{}]: {}".format(i, y_test[i]))
    # Now chunk the val in half
    test_x, val_x, test_y, val_y = train_test_split(x_test, y_test, test_size=0.5, random_state=random_state)
    # print("First few lines of train_x and train_y:")
    # for i in range(5):
        # print("train_x[{}]: {}".format(i, x_train[i]))
        # print("train_y[{}]: {}".format(i, y_train[i]))
    print("Train_x:", train_x.shape)
    print("Train_y:", train_y.shape)
    print("Test_x:", test_x.shape)
    print("Test_y:", test_y.shape)
    print("Val_x:", val_x.shape)
    print("Val_y:", val_y.shape)
    return train_x, train_y, test_x, test_y, val_x, val_y, x_mean, x_std, y_mean, y_std

