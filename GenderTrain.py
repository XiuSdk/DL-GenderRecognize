# AGE
import matplotlib.image as img
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import math
import os
import csv
from GenderUtils import input_data,create_placeholders,random_mini_batches,row_csv2dict,weight_variable,bias_variable,conv2d,max_pool,compute_cost,initialize_parameters,cnn_net,save_model
np.random.seed(1)
tf.reset_default_graph()

def model(X_train, Y_train, X_test, Y_test,learning_rate = 0.001, num_epochs = 100, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 112, 92, 1)
    Y_train -- test set, of shape (None, n_y = 2)
    X_test -- training set, of shape (None, 112, 92, 1)
    Y_test -- test set, of shape (None, n_y = 2)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
#     ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)   
    (m, n_H0, n_W0,n_C0) = X_train.shape   
    n_y = Y_train.shape[1]
    costs = [] 
    SAVE_PATH = "model/mymodel"
    print("X_train shape:",str(X_train.shape))
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    print("Y shape:", str(Y))
    # Initialize parameters
    parameters = initialize_parameters()
    # cnn
    Z3 = cnn_net(X, parameters)
    # Cost function
    cost = compute_cost(Z3, Y)
    # Backpropagation:Define the tensorflow optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    # Inizialize all the variables globally
    init = tf.global_variables_initializer()
    # training process
    saver = tf.train.Saver(max_to_keep=3)
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        # Do the training loop
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X,minibatch_Y) = minibatch
                _,temp_cost = sess.run([optimizer, cost], feed_dict = {X:minibatch_X, Y:minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i : %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        # plot the cost
        #plt.plot(np.squeeze(costs))
        #plt.ylabel("cost")
        #plt.xlabel("iterations (per tens)")
        #plt.title("Lerning ratge =" + str(learning_rate))
        #plt.show()
        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_batch_num = int(math.floor(X_train.shape[0] / minibatch_size))
        train_accuracy = 0.
        for i in range(train_batch_num):
            train_accuracy += 1.0 / train_batch_num * accuracy.eval({X: X_train[i * minibatch_size:(i+1)*minibatch_size,:,:,:],Y:Y_train[i * minibatch_size:(i+1)*minibatch_size,:]})
        test_batch_num = int(X_test.shape[0] / minibatch_size)
        test_accuracy = 0.
        for i in range(test_batch_num):
            test_accuracy += 1.0 / test_batch_num * accuracy.eval({X: X_test[i * minibatch_size:(i+1)*minibatch_size,:,:,:],Y:Y_test[i * minibatch_size:(i+1)*minibatch_size,:]})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        save_model(saver,sess,SAVE_PATH)
        print("Z3's shape:", str(Z3.shape))
        return train_accuracy, test_accuracy, parameters

image_train, label_train, image_test, label_test = input_data()
image_train = image_train.reshape(image_train.shape[0],image_train.shape[1],image_train.shape[2],1)
image_test = image_test.reshape(image_test.shape[0],image_test.shape[1],image_test.shape[2],1)
image_train = image_train / 255.
image_test = image_test / 255.
model(image_train, label_train, image_test, label_test)
