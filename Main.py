import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from sklearn.model_selection import train_test_split

from utils import plot_images,createDataDict,initializeParameters,plot_confusion_matrix
from dataLoad import dataLoad
from singleConnectedNN import fitSimpleNN
from MLP import fitMLP
from convnn import fitCNN

bbs_train,imgs_train,labels = dataLoad()

imgs_size_flat = 60*12

bbs_size_flat = 40*20

imgs_shape = (60, 12)

bbs_shape=(40,20)

num_classes =2

#plot_images(bbs_train[0:9],cls=labels[0:9],img_shape=bbs_shape)

#plot_images(imgs_train[0:9],cls=labels[0:9],img_shape=imgs_shape)

bbs_data = createDataDict(bbs_train,labels)
imgs_data = createDataDict(imgs_train,labels)
print("------------------------------------------------------------------------------------")
print("boundry images")
bbs_pred = []
bbs_acc =[]

parameters = initializeParameters(learning_rate=0.0001, training_epochs=500, batch_size=500,display_size=50)

simple_nn_pred,simple_nn_acc = fitSimpleNN(bbs_size_flat, num_classes, bbs_data,parameters)
layers = [64, 64, 64]
parameters = initializeParameters(learning_rate=0.0075, training_epochs=2000, batch_size=500,display_size=50)
MLP_nn_pred,MLP_nn_acc = fitMLP(layers, bbs_data, bbs_size_flat, num_classes, parameters)
parameters = initializeParameters(learning_rate=1e-4, training_epochs=1000, batch_size=300, display_size=50)
CNN_pred,CNN_acc = fitCNN(bbs_shape, 1, num_classes, bbs_data, parameters)
bbs_pred.append(simple_nn_pred)
bbs_pred.append(CNN_pred)
bbs_pred.append(MLP_nn_pred)
bbs_acc.append(simple_nn_acc)
bbs_acc.append(MLP_nn_acc)
bbs_acc.append(CNN_acc)
bbs_pred = np.array(bbs_pred)
bbs_acc = np.array(bbs_acc)

print("Mean test-set accuracy: {0:.4f}".format(np.mean(bbs_acc)))
print("Min test-set accuracy:  {0:.4f}".format(np.min(bbs_acc)))
print("Max test-set accuracy:  {0:.4f}".format(np.max(bbs_acc)))

ensemble_pred_labels = np.mean(bbs_pred, axis=0)
ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)
ensemble_correct = (ensemble_cls_pred == bbs_data['y_test'])
correct_sum = ensemble_correct.sum()

acc = float(correct_sum) / len(bbs_data['x_test'])
# Print the accuracy.
msg = "Ensenble Accuracy on Test-Set: {0:.1%} ({1} / {2})"
print(msg.format(acc, correct_sum, len(bbs_data['x_test'])))
plot_confusion_matrix(ensemble_cls_pred, bbs_data['y_test'], 2)

print("Best Accuracy: {0:.4f}".format(bbs_acc[np.argmax(bbs_acc)]))







# print("------------------------------------------------------------------------------------")
# print("images")
# parameters = initializeParameters(learning_rate=0.0001, training_epochs=200, batch_size=200)
# fitSimpleNN(imgs_size_flat, num_classes, imgs_data, parameters)











