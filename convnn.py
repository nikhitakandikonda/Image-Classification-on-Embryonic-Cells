import tensorflow as tf
import prettytensor as pt


from utils import plot_image,optimize,print_test_accuracy,initializeParameters,createDataDict,convertToDataset,plot_graphs,plot_confusion_matrix,get_layer_output,get_weights_variable,plot_conv_weights,plot_layer_output,createCheckPoints
from dataLoad import dataLoad

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

total_iterations = 0


def create_model(x_image,training,y_true):
    x_pretty = pt.wrap(x_image)
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer
    norm = pt.BatchNormalizationArguments(scale_after_normalization=True)
    with pt.defaults_scope(activation_fn=tf.nn.elu, phase=phase):
        y_pred, loss = x_pretty. \
            conv2d(kernel=5, depth=64, batch_normalize=norm,name='layer_conv1'). \
            max_pool(kernel=2, stride=2). \
            conv2d(kernel=5, depth=64,name='layer_conv2'). \
            max_pool(kernel=2, stride=2). \
            conv2d(kernel=5, depth=64, l2loss=2, name='layer_conv3'). \
            max_pool(kernel=2, stride=2). \
            flatten(). \
            fully_connected(size=1024,name='layer_fc1'). \
            dropout(keep_prob=0.8, phase=phase). \
            fully_connected(size=512, name='layer_fc1'). \
            softmax_classifier(num_classes=2, labels=y_true)
        return y_pred,loss

def fitCNN(img_shape,num_channels,num_classes,data,parameters):
    x = tf.placeholder(tf.float32, shape=[None, img_shape[0]*img_shape[1]], name='x')
    x_image = tf.reshape(x, [-1, img_shape[0], img_shape[1], num_channels])
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)
    training = True
    with tf.variable_scope('network', reuse=not training):
        _,loss = create_model(x_image,training,y_true)
    optimizer = tf.train.AdamOptimizer(parameters['learning_rate']).minimize(loss)
    training = False
    with tf.variable_scope('network', reuse=not training):
        y_pred,_ = create_model(x_image,training,y_true)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    tfObject = {'x': x, 'y_true': y_true,'y_pred':y_pred, 'optimizer': optimizer, 'y_pred_cls': y_pred_cls, 'accuracy': accuracy,'loss':loss,'saver':saver}
    training_data = convertToDataset(data, parameters['batch_size'])
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path,run_optimize = createCheckPoints(session, saver, parameters['name']+'/cnn')
    parameters['save_path'] = save_path
    if run_optimize:
        optimize(parameters,training_data,data,session,tfObject)
    pred_labels,acc = print_test_accuracy(data,250,tfObject,session)
    weights_conv1 = get_weights_variable(layer_name='layer_conv1')
    weights_conv2 = get_weights_variable(layer_name='layer_conv2')
    weights_conv3 = get_weights_variable(layer_name='layer_conv3')
    output_conv1 = get_layer_output(layer_name='layer_conv1')
    output_conv2 = get_layer_output(layer_name='layer_conv2')
    output_conv3 = get_layer_output(layer_name='layer_conv3')
    plot_conv_weights(weights=weights_conv1,session=session, input_channel=0)
    plot_conv_weights(weights=weights_conv2,session=session, input_channel=0)
    plot_conv_weights(weights=weights_conv3, session=session, input_channel=0)
    img = data['x_test'][10]
    plot_image(img.reshape(img_shape),2008,data['y_test'][10])
    plot_layer_output(output_conv1, img,session,tfObject)
    plot_layer_output(output_conv2, img,session,tfObject)
    plot_layer_output(output_conv3, img, session, tfObject)
    session.close()
    return pred_labels,acc

if __name__=="__main__":
    bbs_train, imgs_train, labels = dataLoad()
    imgs_shape = (60,12)
    bbs_shape = (40,20)
    num_classes = 2
    bbs_data = createDataDict(bbs_train, labels)
    parameters = initializeParameters(name='imgs', learning_rate=1e-4, training_epochs=1000, batch_size=300,
                                      display_size=50)
    fitCNN(bbs_shape,1,num_classes,bbs_data,parameters)
    # imgs_data = createDataDict(imgs_train, labels)
    # parameters = initializeParameters(name='imgs', learning_rate=1e-4, training_epochs=1000, batch_size=300,
    #                                   display_size=50)
    # CNN_pred, CNN_acc = fitCNN(imgs_shape, 1, num_classes, imgs_data, parameters)




