#!/usr/bin/env python3
"""function that trains a loaded neural network
model using mini-batch gradient descent"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        batch_size=32,
        epochs=5,
        load_path="/tmp/model.ckpt",
        save_path="/tmp/model.ckpt"):
    """X_train is the numpy.ndarray containing the training data"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')[0]
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
