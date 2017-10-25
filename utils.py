import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

def plot_images(images, cls, img_shape,cls_pred=None):
    plt.interactive(False)
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape),cmap='jet')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def initializeWeights_Bias(img_size_flat,num_classes):
    weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
    biases = tf.Variable(tf.zeros([num_classes]))
    return weights,biases

def initialize_x_y(img_size_flat,num_classes):
    x = tf.placeholder(tf.float32, [None, img_size_flat])
    y_true = tf.placeholder(tf.float32, [None, num_classes])
    y_true_cls = tf.placeholder(tf.int64, [None])
    return x,y_true,y_true_cls

def initializeParameters(learning_rate,training_epochs,batch_size):
    learning_rate = learning_rate
    training_epochs = training_epochs
    batch_size = batch_size
    parameters ={'learning_rate':learning_rate,'training_epochs':training_epochs,'batch_size':batch_size}
    return parameters

def createDataDict(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    data = {}
    data['x_train']=x_train
    data['x_test'] = x_test
    data['y_train'] = y_train
    data['y_test'] = y_test
    data['y_train_enc'] = oneHotEncoding(y_train)
    data['y_test_enc'] = oneHotEncoding(y_test)
    return data

def oneHotEncoding(labels):
    encLabels = pd.get_dummies(labels).as_matrix()
    return encLabels

def convertToDataset(data,batch_size):
    train_data = tf.contrib.data.Dataset.from_tensor_slices((data['x_train'], data['y_train_enc']))
    train_data = train_data.batch(batch_size)
    train_data = train_data.repeat(100)
    train_data = train_data.shuffle(buffer_size=100,seed=100)# Batch size to use
    train_data_iterator = train_data.make_one_shot_iterator()
    return train_data_iterator

