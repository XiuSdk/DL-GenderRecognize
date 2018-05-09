# AGE
import matplotlib.image as img
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import math
from GenderUtils import create_placeholders,weight_variable,bias_variable,conv2d,max_pool,compute_cost,initialize_parameters,cnn_net
np.random.seed(1)
tf.reset_default_graph()

parameters = initialize_parameters()
saver = tf.train.Saver()
with tf.Session() as sess:
    tf.set_random_seed(1) 
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir = 'model/')
    print(ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
    parameters = {"W1":sess.run(parameters["W1"]),
                 "b1":sess.run(parameters["b1"]),
                 "W2":sess.run(parameters["W2"]),
                 "b2":sess.run(parameters["b2"]),
                 "W3":sess.run(parameters["W3"]),
                 "b3":sess.run(parameters["b3"]),
                 "W4":sess.run(parameters["W4"]),
                 "b4":sess.run(parameters["b4"]),
                 "W5":sess.run(parameters["W5"]),
                 "b5":sess.run(parameters["b5"]),
                 "W6":sess.run(parameters["W6"]),
                 "b6":sess.run(parameters["b6"])}
    #the image inputs is gray image with three channels.
    image = img.imread("data/T3.bmp")
    image_test = image[:,:,0]
    print("image_test shape:", str(image_test.shape))
    image = image_test.reshape(1,image_test.shape[0],image_test.shape[1],1)
#     image = image.reshape(1,image_test.shape[0],image_test.shape[1],1)
    image = image / 255.
    imaget = tf.image.convert_image_dtype(image, tf.float32)
    print("image shape: %", str(imaget.shape))
    res = cnn_net(imaget, parameters)
    print("result: ",sess.run(tf.argmax(res, 1)))
    print(str(res.shape))
    print(res.eval())