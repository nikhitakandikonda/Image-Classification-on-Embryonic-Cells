import tensorflow as tf
from utils import initializeWeights_Bias,initialize_x_y,initializeParameters,oneHotEncoding,createDataDict
from dataLoad import dataLoad

learning_rate, training_epochs, batch_size, display_step = initializeParameters()

def optimize(num_iterations,data,x,y_true,optimizer,session):
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        #total_batch = int(len(data['x_train']) / batch_size)
        # for i in range(total_batch):
        #     batch_xs, batch_ys = data['x_train'].next_batch(batch_size)
        #     feed_dict_train = {x: batch_xs,data['y_trrain']: batch_ys}
        feed_dict_train = {x: data['x_train'], y_true:data['y_train_enc'] }
        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)


def print_accuracy(feed_dict,accuracy,session):
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict)

    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))

def fitSimpleNN(img_size_flat,num_classes,data):
    x, y_true, y_true_cls = initialize_x_y(img_size_flat,num_classes)
    weights,biases = initializeWeights_Bias(img_size_flat, num_classes)

    logits = tf.matmul(x, weights) + biases
    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    feed_dict_test = {x: data['x_test'],
                      y_true: data['y_test_enc'],
                      y_true_cls: data['y_test']}

    optimize(num_iterations=100,data=data,x=x,y_true=y_true,optimizer=optimizer,session=session)
    print_accuracy(feed_dict=feed_dict_test,accuracy=accuracy,session=session)

if __name__=="__main__":
    bbs_train, imgs_train, labels = dataLoad()

    imgs_size_flat = 60 * 12

    bbs_size_flat = 40 * 20

    imgs_shape = (60, 12)

    bbs_shape = (40, 20)

    num_classes = 2

    bbs_data = createDataDict(bbs_train, labels)
    fitSimpleNN(bbs_size_flat, num_classes, bbs_data)





