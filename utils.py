import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math
import itertools
import os
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta


def createCheckPoints(session,saver,model):
    save_dir = 'checkpoints/'+model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir)
    print(save_path)
    try:
        print("Trying to restore last checkpoint ...")

        # Use TensorFlow to find the latest checkpoint - if any.
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

        # Try and load the data in the checkpoint.
        saver.restore(session, save_path=last_chk_path)

        # If we get to this point, the checkpoint was successfully loaded.
        print("Restored checkpoint from:", last_chk_path)
        run_optimize =False
    except:
        # If the above failed for some reason, simply
        # initialize all the variables for the TensorFlow graph.
        print("Failed to restore checkpoint. Initializing variables instead.")
        session.run(tf.global_variables_initializer())
        run_optimize = True
    return save_dir,run_optimize

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


def plot_image(image,i,class_idx):
    script_dir = os.path.dirname(__file__)

    results_dir = os.path.join(script_dir, 'Results\\'+str(class_idx)+'\\')

    # Create figure with sub-plots.
    plt.figure()
    imgplot = plt.imshow(image)
    imgplot.set_cmap('jet')
    plt.colorbar()
    plt.savefig(results_dir+str(i)+'.png')
    plt.close()

def plot_graphs(train_losses,train_acc,val_losses, val_acc,parameters):
    scale = np.arange(0, parameters['training_epochs'], parameters['display_size'])

    plt.subplot(2, 1, 1)
    plt.ylim(min(train_losses) ,max(train_losses))
    plt.plot(scale, train_losses, label='Train')
    plt.plot(scale, val_losses, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(scale, train_acc, label='Train')
    plt.plot(scale, val_acc, label='Validation')
    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

def initializeWeights_Bias(img_size_flat,num_classes):
    weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
    biases = tf.Variable(tf.zeros([num_classes]))
    return weights,biases

def initialize_x_y(img_size_flat,num_classes):
    x = tf.placeholder(tf.float32, [None, img_size_flat])
    y_true = tf.placeholder(tf.float32, [None, num_classes])
    y_true_cls = tf.argmax(y_true, dimension=1)
    return x,y_true,y_true_cls

def initializeParameters(name,learning_rate,training_epochs,batch_size,display_size):
    learning_rate = learning_rate
    training_epochs = training_epochs
    batch_size = batch_size
    display_size =display_size
    parameters ={'name':name,'learning_rate':learning_rate,'training_epochs':training_epochs,'batch_size':batch_size,'display_size':display_size}
    return parameters

def createDataDict(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    x_train,x_val,y_train,y_val = train_test_split(x_train, y_train, test_size=0.10)
    data = {}
    data['x_train']=x_train
    data['x_test'] = x_test
    data['y_train'] = y_train
    data['y_test'] = y_test
    data['x_val'] = x_val
    data['y_val'] = y_val
    data['y_train_enc'] = oneHotEncoding(y_train)
    data['y_test_enc'] = oneHotEncoding(y_test)
    data['y_val_enc'] = oneHotEncoding(y_val)
    return data

def oneHotEncoding(labels):
    encLabels = pd.get_dummies(labels).as_matrix()
    return encLabels

def convertToDataset(data,batch_size):
    train_data = tf.contrib.data.Dataset.from_tensor_slices((data['x_train'], data['y_train_enc']))
    train_data = train_data.batch(batch_size)
    train_data = train_data.repeat(500)
    train_data = train_data.shuffle(buffer_size=100,seed=100)# Batch size to use
    train_data_iterator = train_data.make_one_shot_iterator()
    return train_data_iterator

def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'weights' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.

    with tf.variable_scope("network/" + layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable

def get_layer_output(layer_name):
    # The name of the last operation of the convolutional layer.
    # This assumes you are using Relu as the activation-function.
    tensor_name = "network/" + layer_name + "/Elu:0"

    # Get the tensor with this name.
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

    return tensor



def plot_confusion_matrix(cls_pred,cls_test,num_classes):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.
    class_names=['good','bad']
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))
    title = 'Confusion matrix'
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



def plot_conv_weights(weights,session, input_channel=1):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Print statistics for the weights.
    print("Min:  {0:.5f}, Max:   {1:.5f}".format(w.min(), w.max()))
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)
    abs_max = max(abs(w_min), abs(w_max))

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img,interpolation='nearest', cmap='Greys')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_layer_output(layer_output, image,session,tfOjects):
    # Assume layer_output is a 4-dim tensor
    # e.g. output_conv1 or output_conv2.

    # Create a feed-dict which holds the single input image.
    # Note that TensorFlow needs a list of images,
    # so we just create a list with this one image.
    feed_dict = {tfOjects['x']: [image]}

    # Retrieve the output of the layer after inputting this image.
    values = session.run(layer_output, feed_dict=feed_dict)

    # Get the lowest and highest values.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    values_min = np.min(values)
    values_max = np.max(values)

    # Number of image channels output by the conv. layer.
    num_images = values.shape[3]

    # Number of grid-cells to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_images))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid image-channels.
        if i < num_images:
            # Get the images for the i'th output channel.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, vmin=values_min, vmax=values_max,
                      interpolation='nearest', cmap='jet')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def print_test_accuracy(data,test_batch_size,tfObjects,session):

    # Number of images in the test-set.
    num_test = len(data['x_test'])
    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    pred_labels = np.zeros(shape=(num_test, 2),
                           dtype=np.float)

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
        pred_labels[i:j]=session.run(tfObjects['y_pred'], feed_dict=feed_dict)
        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(tfObjects['y_pred_cls'], feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data['y_test']
    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)
    #plot images misclassifed
    # for idx,classification in enumerate(correct):
    #     if not classification:
    #         print(idx)
    #         print(str(cls_true[idx])+"------------->"+str(cls_pred[idx]))
    #         plot_image(data['x_test'][idx].reshape(60,12),idx,cls_true[idx])
    #         # output_conv2 = get_layer_output(layer_name='layer_conv3')
    #         # plot_layer_output(output_conv2, data['x_test'][idx], session, tfObjects)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test
    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    plot_confusion_matrix(cls_pred,cls_true,2)
    return pred_labels,acc

def optimize(parameters,training_data,data,session,tfObjects):
    train_losses = list()
    train_accs = list()
    val_losses = list()
    val_accs = list()
    # Start-time used for printing time-usage below.
    start_time = time.time()
    x_val = data['x_val']
    y_val = data['y_val_enc']
    for i in range(parameters['training_epochs']):

        x_y_batch = training_data.get_next()
        x_y_batch = session.run(x_y_batch)

        x_batch, y_true_batch = x_y_batch[0], x_y_batch[1]


        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {tfObjects['x']: x_batch,
                           tfObjects['y_true']: y_true_batch}

        feed_dict_val = {tfObjects['x']: x_val,
                           tfObjects['y_true']: y_val}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(tfObjects['optimizer'], feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % parameters['display_size'] == 0:
            # Calculate the accuracy on the training-set.
            train_loss,train_acc = session.run([tfObjects['loss'],tfObjects['accuracy']], feed_dict=feed_dict_train)
            val_loss, val_acc = session.run([tfObjects['loss'],tfObjects['accuracy']], feed_dict=feed_dict_val)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.4%},Validation Accuracy: {2:>6.4%},Training Loss: {3:>6.4},Validation Loss : {4:>6.4}"

            # Print it.
            print(msg.format(i + 1, train_acc,val_acc,train_loss,val_loss))

    tfObjects['saver'].save(session,
               save_path=parameters['save_path'])

    print("Saved checkpoint.")

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    plot_graphs(train_losses, train_accs, val_losses, val_accs, parameters)
