import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from sklearn.model_selection import train_test_split

from utils import plot_images,createDataDict,initializeParameters
from dataLoad import dataLoad
from singleConnectedNN import fitSimpleNN

bbs_train,imgs_train,labels = dataLoad()

imgs_size_flat = 60*12

bbs_size_flat = 40*20

imgs_shape = (60, 12)

bbs_shape=(40,20)

num_classes =2

plot_images(bbs_train[0:9],cls=labels[0:9],img_shape=bbs_shape)

plot_images(imgs_train[0:9],cls=labels[0:9],img_shape=imgs_shape)

bbs_data = createDataDict(bbs_train,labels)
imgs_data = createDataDict(imgs_train,labels)
print("------------------------------------------------------------------------------------")
print("boundry images")
parameters = initializeParameters(learning_rate=0.0001, training_epochs=200, batch_size=200)
fitSimpleNN(bbs_size_flat, num_classes, bbs_data,parameters)
print("------------------------------------------------------------------------------------")
print("images")
parameters = initializeParameters(learning_rate=0.0001, training_epochs=200, batch_size=200)
fitSimpleNN(imgs_size_flat, num_classes, imgs_data, parameters)









