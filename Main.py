import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

bbs_train_file = open('training_demo/bbs-train.txt', 'r')
bbs_train = bbs_train_file.readlines()

imgs_train_file = open('training_demo/imgs-train.txt', 'r')
imgs_train = imgs_train_file.readlines()

label_file = open('training_demo/label-train.txt', 'r')
label = label_file.readlines()



