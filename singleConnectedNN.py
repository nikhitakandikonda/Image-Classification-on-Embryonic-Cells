import tensorflow as tf
from utils import createCheckPoints,optimize,print_test_accuracy,initializeWeights_Bias,initialize_x_y,initializeParameters,createDataDict,convertToDataset,plot_confusion_matrix
from dataLoad import dataLoad

def fitSimpleNN(img_size_flat,num_classes,data,parameters):
    x, y_true, y_true_cls = initialize_x_y(img_size_flat, num_classes)
    weights, biases = initializeWeights_Bias(img_size_flat, num_classes)

    logits = tf.matmul(x, weights) + biases
    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=y_true)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdagradOptimizer(parameters['training_epochs']).minimize(loss)


    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    tfObject = {'x': x, 'y_true': y_true, 'optimizer': optimizer, 'y_pred_cls': y_pred_cls, 'accuracy': accuracy,
                'loss': loss, 'saver': saver}
    train_data = convertToDataset(data,parameters['batch_size'])

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path, run_optimize = createCheckPoints(session, saver, 'Logistic Reg')
    parameters['save_path'] = save_path
    if run_optimize:
        optimize(parameters,train_data,data,tfObjects=tfObject,session = session)

    pred_labels,acc = print_test_accuracy(data, 250, tfObject, session)
    session.close()
    return pred_labels,acc

if __name__=="__main__":
    bbs_train, imgs_train, labels = dataLoad()
    imgs_size_flat = 60 * 12
    bbs_size_flat = 40 * 20
    num_classes = 2
    # bbs_data = createDataDict(bbs_train, labels)
    # parameters = initializeParameters(learning_rate=0.0001, training_epochs=500, batch_size=500,display_size=50)
    # fitSimpleNN(bbs_size_flat, num_classes, bbs_data,parameters)
    imgs_data = createDataDict(imgs_train,labels)
    parameters = initializeParameters(learning_rate=0.0001, training_epochs=600, batch_size=500,display_size=50)
    fitSimpleNN(imgs_size_flat, num_classes, imgs_data, parameters)



