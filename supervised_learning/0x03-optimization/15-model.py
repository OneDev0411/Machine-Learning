#!/usr/bin/env python3
"""model"""
import tensorflow as tf
import numpy as np


def create_placeholders(nx, classes):
    """nx: the number of feature columns in our data
    classes: the number of classes in our classifier"""
    x = tf.placeholder("float", [None, nx], "x")
    y = tf.placeholder("float", [None, classes], "y")
    return x, y

def create_layer(prev, n, activation):
    """prev is the tensor output of the previous layer
    n is the number of nodes in the layer to create
    activation is the activation function that the layer should use"""
    layer = tf.layers.Dense(
        n,
        activation,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"),
        name="Layer")
    return layer(prev)


def forward_prop(x, layer_sizes=[], activations=[], epsilon=1e-8):
    """x is the placeholder for the input data
    layer_sizes is a list containing the number of nodes in each layer
    activations is a list containing the activation functions for each layer
    Returns: the prediction of the network in tensor form"""
    for i in range(len(layer_sizes)):
        if i == len(layer_sizes) - 1:
            layer = create_layer(x, layer_sizes[i], activations[i])
            x = layer
        else:
            x = create_batch_norm_layer(x, layer_sizes[i], activations[i], epsilon)
    return x


def calculate_accuracy(y, y_pred):
    """y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions"""
    accuracy = tf.math.reduce_mean(
        tf.cast(tf.equal(
            tf.argmax(y, axis=-1),
            tf.argmax(y_pred, axis=-1)), tf.float32))
    return accuracy


def calculate_loss(y, y_pred):
    """y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions"""
    return tf.losses.softmax_cross_entropy(y, y_pred)


def shuffle_data(X, Y):
    """X is the numpy.ndarray of shape (m, nx) to shuffle
    Y is the second numpy.ndarray of shape (m, ny) to shuffle"""
    p = np.random.permutation(len(X))
    return X[p], Y[p]


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ that creates the training operation for a neural network
    in tensorflow using the Adam optimization algorithm"""
    return tf.train.AdamOptimizer(
        alpha, beta1, beta2, epsilon=epsilon).minimize(loss)


def create_batch_norm_layer(prev, n, activation, epsilon):
    """function that that creates a batch
    normalization layer for a neural network in tensorflow"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, name="layer",
                            kernel_initializer=kernel)
    mean, variance = tf.nn.moments(layer(prev), [0])
    gamma = tf.ones([n])
    beta = tf.zeros([n])
    norm = tf.nn.batch_normalization(
        layer(prev), mean, variance, beta, gamma, epsilon)
    return activation(norm)


def model(
        Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
        beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32,
        epochs=5, save_path='/tmp/model.ckpt'):
    """function that builds, trains, and saves a
    neural network model in tensorflow using
    Adam optimization, mini-batch gradient descent,
    learning rate decay, and batch normalization"""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layers, activations, epsilon)
    loss = calculate_loss(y=y, y_pred=y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)
    init = tf.initializers.global_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        for i in range(epochs + 1):
            X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)
            tLoss = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            tAccuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            vLoss = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            vAccuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(tLoss))
            print("\tTraining Accuracy: {}".format(tAccuracy))
            print("\tValidation Cost: {}".format(vLoss))
            print("\tValidation Accuracy: {}".format(vAccuracy))
            if i != epochs:
                batch = X_train.shape[0] // batch_size
                if batch % batch_size != 0:
                    batch += 1
                    batcher = 1
                else:
                    batcher = 0
                for j in range(batch):
                    start = j * batch_size
                    if j == batch - 1 and batcher == 1:
                        end = X_train.shape[0]
                    else:
                        end = j * batch_size + batch_size
                    X_batch = X_shuffle[start:end]
                    Y_batch = Y_shuffle[start:end]
                    sess.run(train_op, {x: X_batch, y: Y_batch})
                    loss_train = sess.run(loss, {x: X_batch, y: Y_batch})
                    acc_train = sess.run(accuracy, {x: X_batch, y: Y_batch})
                    if (j+1) % 100 == 0 and j != 0:
                        print('\tStep {}:'.format(j+1))
                        print('\t\tCost: {}'.format(loss_train))
                        print('\t\tAccuracy: {}'.format(acc_train))
        return saver.save(sess, save_path)
