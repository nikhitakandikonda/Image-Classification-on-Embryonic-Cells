import tensorflow as tf
from utils import initializeWeights_Bias,initialize_x_y,initializeParameters,createDataDict,convertToDataset
from dataLoad import dataLoad


def print_accuracy(feed_dict,accuracy,session):
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict)

    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))

def neural_net(x,weights,biases,num_layers):
    # Hidden fully connected layer with 256 neurons
    a = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    out = tf.nn.relu(a)
    for i in range(1,num_layers):
        a =tf.add(tf.matmul(out, weights['h'+str(i+1)]), biases['b'+str(i+1)])
        out = tf.nn.relu(a)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(out, weights['out']) + biases['out']
    return out_layer

def optimize(num_iterations,session,tfObjects,training_data):
    for step in range(1, num_iterations+1):
        x_y_batch = training_data.get_next()
        x_y_batch = session.run(x_y_batch)

        x_batch, y_true_batch = x_y_batch[0], x_y_batch[1]
        # Run optimization op (backprop)
        session.run(tfObjects['train_op'], feed_dict={tfObjects['X']: x_batch, tfObjects['Y_true']: y_true_batch})
        if step % 10 == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = session.run([tfObjects['loss_op'],tfObjects['accuracy']], feed_dict={tfObjects['X']: x_batch, tfObjects['Y_true']: y_true_batch})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")



def fitMLP(layers,data,img_size_flat,num_classes,parameters):
    X, Y_true, Y_true_cls = initialize_x_y(img_size_flat, num_classes)
    num_layers = len(layers)
    weights = {'h1': tf.Variable(tf.truncated_normal([img_size_flat, layers[0]])),
               'out': tf.Variable(tf.truncated_normal([layers[-1], num_classes]))}
    biases ={'b1': tf.Variable(tf.zeros([layers[0]])),
            'out': tf.Variable(tf.zeros([num_classes]))}
    for i in range(1,num_layers):
        weights['h'+str(i+1)]= tf.Variable(tf.truncated_normal([layers[i-1], layers[i]]))
        biases['b'+str(i+1)] = tf.Variable(tf.zeros([layers[i]]))

    logits = neural_net(X,weights,biases,num_layers)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y_true))
    optimizer = tf.train.AdamOptimizer(learning_rate=parameters['learning_rate'])
    train_op = optimizer.minimize(loss_op)

    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    train_data_iterator = convertToDataset(data, parameters['batch_size'])
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    feed_dict_test = {X: data['x_test'],
                      Y_true: data['y_test_enc'],
                      Y_true_cls: data['y_test']}
    tfObjects = {'X':X,'Y_true':Y_true,'accuracy':accuracy,'train_op':train_op,'loss_op':loss_op}
    optimize(parameters['training_epochs'], session, tfObjects, train_data_iterator)
    print_accuracy(feed_dict=feed_dict_test, accuracy=accuracy, session=session)


if __name__=="__main__":
    bbs_train, imgs_train, labels = dataLoad()
    imgs_size_flat = 60 * 12
    bbs_size_flat = 40 * 20
    num_classes = 2
    bbs_data = createDataDict(bbs_train, labels)
    layers =[256,256,200,100]
    parameters = initializeParameters(learning_rate=0.005, training_epochs=500, batch_size=100)
    fitMLP(layers,bbs_data,bbs_size_flat, num_classes ,parameters)
    imgs_data = createDataDict(imgs_train,labels)
    parameters = initializeParameters(learning_rate=0.005, training_epochs=1000, batch_size=100)
    fitMLP(layers,imgs_data,imgs_size_flat, num_classes, parameters)