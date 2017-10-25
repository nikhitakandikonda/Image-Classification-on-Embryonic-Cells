import tensorflow as tf
import prettytensor as pt
import os
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import numpy as np

from utils import initializeWeights_Bias,initialize_x_y,initializeParameters,createDataDict,convertToDataset
from dataLoad import dataLoad

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

total_iterations = 0

def optimize(num_iterations,training_data,session,tfObjects):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        x_y_batch = training_data.get_next()
        x_y_batch = session.run(x_y_batch)

        x_batch, y_true_batch = x_y_batch[0], x_y_batch[1]


        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {tfObjects['x']: x_batch,
                           tfObjects['y_true']: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(tfObjects['optimizer'], feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(tfObjects['accuracy'], feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def print_test_accuracy(data,test_batch_size,tfObjects,session):

    # Number of images in the test-set.
    num_test = len(data['x_test'])

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data['x_test'][i:j, :]

        # Get the associated labels.
        labels = data['y_test_enc'][i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {tfObjects['x']: images,
                     tfObjects['y_true']: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(tfObjects['y_pred_cls'], feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data['y_test']

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))



def fitCNN(img_shape,num_channels,num_classes,data):
    x = tf.placeholder(tf.float32, shape=[None, img_shape[0]*img_shape[1]], name='x')
    x_image = tf.reshape(x, [-1, img_shape[0], img_shape[1], num_channels])
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    x_pretty = pt.wrap(x_image)

    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred, loss = x_pretty. \
            conv2d(kernel=5, depth=30, name='layer_conv1'). \
            max_pool(kernel=2, stride=2). \
            conv2d(kernel=5, depth=36, name='layer_conv2'). \
            max_pool(kernel=2, stride=2). \
            flatten(). \
            fully_connected(size=128, name='layer_fc1'). \
            softmax_classifier(num_classes=num_classes, labels=y_true)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tfObject = {'x': x, 'y_true': y_true, 'optimizer': optimizer, 'y_pred_cls': y_pred_cls, 'accuracy': accuracy}
    train_data_iterator = convertToDataset(data, 100)
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    optimize(900,train_data_iterator,session,tfObject)
    print_test_accuracy(data,250,tfObject,session)
    session.close()


if __name__=="__main__":
    bbs_train, imgs_train, labels = dataLoad()
    imgs_shape = (60,12)
    bbs_shape = (40,20)
    num_classes = 2
    bbs_data = createDataDict(bbs_train, labels)
    #parameters = initializeParameters(learning_rate=0.005, training_epochs=500, batch_size=100)
    fitCNN(bbs_shape,1,num_classes,bbs_data)




