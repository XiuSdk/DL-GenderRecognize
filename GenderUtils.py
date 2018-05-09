# AGE
import matplotlib.image as img
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import math
import os
import csv

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(name='X', shape=(None, n_H0, n_W0, n_C0), dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=(None, n_y), dtype=tf.float32)    
    return X, Y

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m / mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, int(num_complete_minibatches)):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def row_csv2dict(csv_file):
    dict_club={}
    with open(csv_file)as f:
        reader=csv.reader(f,delimiter=',')
        for row in reader:
            dict_club[row[0]]=row[1]
    return dict_club

def input_data():
    
    path = "data/train/"
    train_num = sum([len(x) for _, _, x in os.walk(os.path.dirname(path))])
    image_train = np.zeros((train_num,112,92))
    label_train = np.ones((train_num,2))
    train_label_dict = row_csv2dict("data/train.csv")
    count = 0
    for key in train_label_dict:
        if int(train_label_dict[key]) == 0:
            label_train[count, 0] = 1
            label_train[count, 1] = 0
        else:
            label_train[count, 1] = 1
            label_train[count, 0] = 0
        filename = path + str(key)
        image_train[count] = img.imread(filename)
        count = count + 1
    path = "data/test/" 
    test_num = sum([len(x) for _, _, x in os.walk(os.path.dirname(path))])
    image_test = np.zeros((test_num, 112,92))
    label_test = np.ones((test_num,2))
    test_label_dict = row_csv2dict("data/test.csv")
    count = 0
    for key in test_label_dict:
        if int(test_label_dict[key]) == 0:
            label_test[count, 0] = 1
            label_test[count, 1] = 0
        else:
            label_test[count, 1] = 1
            label_test[count, 0] = 0
        filename = path + str(key)
        image_test[count] = img.imread(filename)
        count = count + 1
    return image_train, label_train,image_test, label_test

def weight_variable(shape,name):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1),name=name)

def bias_variable(shape,name):
    return tf.Variable(tf.constant(0.1, shape = shape),name=name)

def conv2d(x,w,padding="SAME"):
    if padding=="SAME" :
        return tf.nn.conv2d(x, w, strides = [1,1,1,1], padding = "SAME")
    else:
        return tf.nn.conv2d(x, w, strides = [1,1,1,1], padding = "VALID")
    
def max_pool(x, kSize, Strides):
    return tf.nn.max_pool(x, ksize = [1,kSize,kSize,1],strides = [1,Strides,Strides,1], padding = "SAME")    

def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))    
    return cost

def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.cast(weight_variable([5,5,1,32],"W1"), dtype = tf.float32)
    b1 = tf.cast(bias_variable([32],"b1"), dtype = tf.float32)
    W2 = tf.cast(weight_variable([5,5,32,64],"W2"), dtype = tf.float32)
    b2 = tf.cast(bias_variable([64],"b2"), dtype = tf.float32)
    W3 = tf.cast(weight_variable([5,5,64,128],"W3"), dtype = tf.float32)
    b3 = tf.cast(bias_variable([128],"b3"), dtype = tf.float32)
    
    W4 = tf.cast(weight_variable([14*12*128,500],"W4"), dtype = tf.float32)
    b4 = tf.cast(bias_variable([500],"b4"), dtype = tf.float32)
    W5 = tf.cast(weight_variable([500,500],"W5"), dtype = tf.float32)
    b5 = tf.cast(bias_variable([500],"b5"), dtype = tf.float32)
    W6 = tf.cast(weight_variable([500,2],"W6"), dtype = tf.float32)
    b6 = tf.cast(bias_variable([2],"b6"), dtype = tf.float32)
    parameters = {"W1":W1,
                 "b1":b1,
                 "W2":W2,
                 "b2":b2,
                 "W3":W3,
                 "b3":b3,
                 "W4":W4,
                 "b4":b4,
                 "W5":W5,
                 "b5":b5,
                 "W6":W6,
                 "b6":b6}
    return parameters

def cnn_net(x, parameters, keep_prob = 1.0):
    #frist convolution layer
    w_conv1 = parameters["W1"]
    b_conv1 = parameters["b1"]
    h_conv1 = tf.nn.relu(conv2d(x,w_conv1) + b_conv1)  #output size 112x92x32
    h_pool1 = max_pool(h_conv1,2,2)    #output size 56x46x32
    
    #second convolution layer
    w_conv2 = parameters["W2"]
    b_conv2 = parameters["b2"]
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2) #output size 56x46x64
    h_pool2 = max_pool(h_conv2,2,2) #output size 28x23x64
    
    #third convolution layer
    w_conv3 = parameters["W3"]
    b_conv3 = parameters["b3"]
    h_conv3 = tf.nn.relu(conv2d(h_pool2,w_conv3) + b_conv3) #output size 28x23x128
    h_pool3 = max_pool(h_conv3,2,2) #output size 14x12x128
    
    #full convolution layer 
    w_fc1 = parameters["W4"]
    b_fc1 = parameters["b4"]
    h_fc11 = tf.reshape(h_pool3,[-1,14*12*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_fc11,w_fc1) + b_fc1)
    
    w_fc2 = parameters["W5"]
    b_fc2 = parameters["b5"]
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1,w_fc2)+b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)
    
    w_fc3 = parameters["W6"]
    b_fc3 = parameters["b6"]
    y_conv = tf.matmul(h_fc2_drop, w_fc3) + b_fc3
    #y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, w_fc3) + b_fc3)
    #rmse = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_conv))
    #train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    #correct_prediction  = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return y_conv

def save_model(saver,sess,save_path):
    path = saver.save(sess, save_path)
    print 'model save in :{0}'.format(path)

