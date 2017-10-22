import tensorflow as tf
from utils import initializeWeights_Bias,initialize_x_y,initializeParameters,createDataDict,convertToDataset
from dataLoad import dataLoad

def print_accuracy(feed_dict,accuracy,session):
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict)

    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))

def optimize(num_iterations,training_data,tfObjects,session):
    for epoch in range(num_iterations):
        avg_cost = 0.
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_y_batch = training_data.get_next()
        x_y_batch = session.run(x_y_batch)

        x_batch, y_true_batch = x_y_batch[0],x_y_batch[1]

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training.
        feed_dict_train = {tfObjects['x']: x_batch,
                           tfObjects['y_true']: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        optimizer = tfObjects['optimizer']
        cost = tfObjects['cost']
        _,c = session.run([optimizer,cost], feed_dict=feed_dict_train)
        avg_cost += c / x_batch.shape[0]
        if (epoch + 1) % 10 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))



def fitSimpleNN(img_size_flat,num_classes,data,parameters):
    x, y_true, y_true_cls = initialize_x_y(img_size_flat, num_classes)
    weights, biases = initializeWeights_Bias(img_size_flat, num_classes)

    logits = tf.matmul(x, weights) + biases
    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.GradientDescentOptimizer(parameters['learning_rate']).minimize(cost)


    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tfObject = {'x': x, 'y_true': y_true, 'optimizer': optimizer, 'cost': cost, 'accuracy': accuracy}
    feed_dict_test = {x: data['x_test'],
                      y_true: data['y_test_enc'],
                      y_true_cls: data['y_test']}

    train_data_iterator = convertToDataset(data,parameters['batch_size'])

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    optimize(num_iterations=parameters['training_epochs'],training_data=train_data_iterator,tfObjects=tfObject,session = session)
    print_accuracy(feed_dict=feed_dict_test, accuracy=accuracy, session=session)


if __name__=="__main__":
    bbs_train, imgs_train, labels = dataLoad()
    imgs_size_flat = 60 * 12
    bbs_size_flat = 40 * 20
    num_classes = 2
    bbs_data = createDataDict(bbs_train, labels)
    parameters = initializeParameters(learning_rate=0.001, training_epochs=150, batch_size=500)
    fitSimpleNN(bbs_size_flat, num_classes, bbs_data,parameters)
    imgs_data = createDataDict(imgs_train,labels)
    parameters = initializeParameters(learning_rate=0.001, training_epochs=200, batch_size=500)
    fitSimpleNN(imgs_size_flat, num_classes, imgs_data, parameters)



