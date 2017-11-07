import tensorflow as tf
from utils import print_test_accuracy,createCheckPoints,optimize,initialize_x_y,initializeParameters,createDataDict,convertToDataset
from dataLoad import dataLoad



def neural_net(x,weights,biases,num_layers):
    # Hidden fully connected layer with 256 neurons
    a = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    out = tf.nn.elu(a)
    for i in range(1,num_layers):
        a =tf.add(tf.matmul(out, weights['h'+str(i+1)]), biases['b'+str(i+1)])
        out = tf.nn.elu(a)
        tf.nn.dropout(out,keep_prob=0.8)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(out, weights['out']) + biases['out']
    return out_layer


def fitMLP(layers,data,img_size_flat,num_classes,parameters):
    x, y_true, y_true_cls = initialize_x_y(img_size_flat, num_classes)
    num_layers = len(layers)
    weights = {'h1': tf.Variable(tf.truncated_normal([img_size_flat, layers[0]])),
               'out': tf.Variable(tf.truncated_normal([layers[-1], num_classes]))}
    biases ={'b1': tf.Variable(tf.zeros([layers[0]])),
            'out': tf.Variable(tf.zeros([num_classes]))}
    for i in range(1,num_layers):
        weights['h'+str(i+1)]= tf.Variable(tf.truncated_normal([layers[i-1], layers[i]]))
        biases['b'+str(i+1)] = tf.Variable(tf.zeros([layers[i]]))

    logits = neural_net(x,weights,biases,num_layers)
    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(y_pred, dimension=1)


    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y_true))
    optimizer = tf.train.AdagradOptimizer(learning_rate=parameters['learning_rate']).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    tfObject = {'x': x, 'y_true': y_true, 'optimizer': optimizer,'y_pred':y_pred, 'y_pred_cls': y_pred_cls, 'accuracy': accuracy,
                'loss': loss, 'saver': saver}

    train_data = convertToDataset(data, parameters['batch_size'])

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path, run_optimize = createCheckPoints(session, saver, parameters['name']+'/MLP')
    parameters['save_path'] = save_path
    if run_optimize:
        optimize(parameters, train_data, data, tfObjects=tfObject, session=session)
    pred_labels,acc = print_test_accuracy(data, 250, tfObject, session)
    session.close()
    return  pred_labels,acc


if __name__=="__main__":
    bbs_train, imgs_train, labels = dataLoad()
    imgs_size_flat = 60 * 12
    bbs_size_flat = 40 * 20
    num_classes = 2
    bbs_data = createDataDict(bbs_train, labels)
    layers =[64,64,64]
    parameters = initializeParameters(learning_rate=0.005, training_epochs=2500, batch_size=500,display_size=50)
    fitMLP(layers,bbs_data,bbs_size_flat, num_classes ,parameters)
    # imgs_data = createDataDict(imgs_train,labels)
    # parameters = initializeParameters(learning_rate=0.005, training_epochs=1000, batch_size=100,display_size=50)
    # fitMLP(layers,imgs_data,imgs_size_flat, num_classes, parameters)