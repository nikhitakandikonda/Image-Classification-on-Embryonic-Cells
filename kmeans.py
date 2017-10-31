from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
from utils import convertToDataset,createDataDict
from dataLoad import dataLoad

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


bbs_train, imgs_train, labels = dataLoad()

imgs_size_flat = 60*12

bbs_size_flat = 40*20

imgs_shape = (60, 12)

bbs_shape=(40,20)

num_classes =2

num_steps = 50 # Total steps to train
batch_size = 1024 # The number of samples per batch
k = 25

bbs_data = createDataDict(bbs_train,labels)
imgs_data = createDataDict(imgs_train,labels)
training_data_bbs = convertToDataset(bbs_data, batch_size)


X = tf.placeholder(tf.float32, shape=[None, bbs_size_flat])
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)
(all_scores, cluster_idx, scores, cluster_centers_initialized, init_op,
train_op) = kmeans.training_graph()
cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
avg_distance = tf.reduce_mean(scores)

init_vars = tf.global_variables_initializer()

# Start TensorFlow session
session = tf.Session()
session.run(tf.global_variables_initializer())


x_y_batch = training_data.get_next()
x_y_batch = session.run(x_y_batch)

# Run the initializer
sess.run(init_vars, feed_dict={X: bbs_data['x_train']})
sess.run(init_op, feed_dict={X: bbs_data['x_train']})

for i in range(1, num_steps + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: full_data_x})
    if i % 10 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))



