"""

@Author: Waybarrios
13 November 2015 10:04 AM

"""
#Import useful libs

import tensorflow as tf
import numpy as np
import input_data as data


"""

MNIST URL: http://yann.lecun.com/exdb/mnist/

This tutorial is intended for readers who are new to both machine learning and TensorFlow. 
you already know what MNIST is, and what softmax (multinomial logistic) regression is, you might prefer this faster paced tutorial.

When one learns how to program, there's a tradition that the first thing you do is print "Hello World." 
Just like programming has Hello World, machine learning has MNIST.

"""
#Downloading MNIST dataset

mnist = data.read_data_sets("MNIST_data/", one_hot=True)

"""

Tensorflow relies on a highly efficient C++ backend to do its computation. The connection to this backend is called a session. 
The common usage for TensorFlow programs is to first create a graph and then launch it in a session.


Here we instead use the convenience InteractiveSession class, which makes TensorFlow more flexible about how you structure your code. 
It allows you to interleave operations which build a computation graph with ones that run the graph. 
This is particularly convenient when working in interactive contexts like iPython. If you are not using an InteractiveSession, 
then you should build the entire computation graph before starting a session and launching the graph.


"""

#Launching Interactive session
sess = tf.InteractiveSession()


"""
--------------------------------------------------------------------
Training function
-------------------------------------------------------------------

"""
def training_model_softmax (x,y_,y,cross_entropy,):
	print "--------------------------------------------------------------------"
	print "Training Softmax model  "
	print "--------------------------------------------------------------------"
	#We will use steepest gradient descent, with a step length of 0.01, to descend the cross entropy.
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

	#Check accuracy
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


	#The returned operation train_step, when run, will apply the gradient descent updates to the parameters. 
	for i in range(1000):
		
  		batch = mnist.train.next_batch(50)
  		train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  		train_accuracy = accuracy.eval(feed_dict={
	        x:batch[0], y_: batch[1]})
  		print "step %d, training accuracy %g"%(i, train_accuracy)
	  	

  	print "--------------------------------------------------------------------"
	print "Finishied Tranining Softmax model  "
	print "--------------------------------------------------------------------"


"""
--------------------------------------------------------------------
Evaluating function
-------------------------------------------------------------------

"""
def evaluating_model_softmax (x,ymodel,yreal):
	#Check if our prediction matches the truth
	correct_prediction = tf.equal(tf.argmax(ymodel,1), tf.argmax(yreal,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	return accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})


"""
--------------------------------------------------------------------
Softmax function
-------------------------------------------------------------------

"""


def build_softmax(x,y_):

	"""
	--------------------------------------------------------------------------------------------

	Build a Softmax Regression Model 

	--------------------------------------------------------------------------------------------
	"""
	
	"""
	Here x and y_ aren't specific values. Rather, they are each a placeholder -- a 
	value that we'll input when we ask TensorFlow to run a computation.

	Here we assign it a shape of [None, 784], where 784 is the dimensionality of a single flattened MNIST image. None means any size.

	28x28 = 784 (grayscale images)
	10 = Number of classes
	"""

	#Creating model parameters
	W = tf.Variable (tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))

	"""
	W is a 784x10 matrix (because we have 784 input features and 10 outputs) and b is a 10-dimensional vector (because we have 10 classes).
	"""

	#Initiliaze variables within interactive session.

	sess.run(tf.initialize_all_variables())

	#Softmax model

	y = tf.nn.softmax(tf.matmul(x,W) + b)

	#Cost function to be minized during training (our cost function will be cross entropy)

	cross_entropy = -tf.reduce_sum(y_*tf.log(y))

	return cross_entropy, y

"""
--------------------------------------------------------------------
Create Convolutional Neuronal Networks variables

NOTE: One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients
--------------------------------------------------------------------

"""
def weight_variable (shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable (shape):
	initial = tf.constant(0.1, shape=shape)
  	return tf.Variable(initial)


"""
--------------------------------------------------------------------
Convolution and Pooling operations 

Stride size= Vanilla Version
Convolution uses a stride one and are zero padded (output is same as that input)
Pooling uses 2x2 block kernel 
-------------------------------------------------------------------

"""

def conv2d (x,W): 
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


"""
--------------------------------------------------------------------
Multilayer Convolutional Network 
-------------------------------------------------------------------

"""

def build_CNN (x):

	"""
	--------------------------------------------------------------------
	First Convolutional layer 
	-------------------------------------------------------------------

	"""
	# (32 features for each 5x5 patch) 32 lenght of output

	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	"""

	To apply the layer, 
	we first reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height, 
	and the final dimension corresponding to the number of color channels.

	"""

	x_image = tf.reshape(x, [-1,28,28,1])

	#We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
	#ReLU function 
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	#max pooling
	h_pool1 = max_pool_2x2(h_conv1)

	"""
	--------------------------------------------------------------------
	Second Convolutional layer 
	-------------------------------------------------------------------

	"""
	#Second layer will have 64 features for each 5x5 patch
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	#ReLU
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	#Max pooling
	h_pool2 = max_pool_2x2(h_conv2)

	"""
	--------------------------------------------------------------------
	Fully Connected layer  (Densely Connected)
	-------------------------------------------------------------------

	"""
	#Image size has been reduced 7x7
	#Fully connected layer with 1024 neurons

	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	"""
	--------------------------------------------------------------------
	Dropout 
	-------------------------------------------------------------------

	"""
	#Reduce overfitting we apply dropout before softmax layer 
	#tf.nn.dropout op automatically handles scaling neuron outputs 
	#in addition to masking them, so dropout just works without any additional scaling.
	keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


	"""
	--------------------------------------------------------------------
	Softmax Layer
	-------------------------------------------------------------------

	"""
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])

	#Applpying softmax
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	return y_conv, keep_prob

	"""
	--------------------------------------------------------------------
	Training and Evaluating
	-------------------------------------------------------------------

	"""

def train_and_evaluate (x,y_conv,y_,keep_prob) :

	"""
	we will replace the steepest gradient descent optimizer with 
	the more sophisticated ADAM optimizer; 
	we will include the additional parameter keep_prob in feed_dict to control the dropout rate; and 
	we will add logging to every 100th iteration in the training process.
	"""
	cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	sess.run(tf.initialize_all_variables())

	print "--------------------------------------------------------------------"
	print "Training Convolutional Neuronal Networks model  "
	print "--------------------------------------------------------------------"

	for i in range(20000) :
	  batch = mnist.train.next_batch(50)
	  if i%100 == 0:
	    train_accuracy = accuracy.eval(feed_dict={
	        x:batch[0], y_: batch[1], keep_prob: 1.0})
	    print "step %d, training accuracy %g"%(i, train_accuracy)
	  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	print "--------------------------------------------------------------------"
	print "Finished Training Convolutional Neuronal Networks model  "
	print "--------------------------------------------------------------------"
	
	return accuracy.eval(feed_dict={
	    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

	print "test accuracy %g"%accuracy.eval(feed_dict={
	    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})


#We start building the computation graph by creating nodes for the input images and target output classes

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

#Build softmax model
cross_entropy, y= build_softmax(x, y_)

#Training softmax model
training_model_softmax (x,y_,y,cross_entropy)

#Evaluating softmax model
soft_accuray = evaluating_model_softmax (x,y,y_)

#Build CNN
y_conv, keep_prob = build_CNN(x)

cnn_accuray = train_and_evaluate (x,y_conv,y_,keep_prob)


print "--------------------------------------------------------------------"
print " TESTING DATA  "
print "--------------------------------------------------------------------"
	
	
print "Accuracy using Softmax model %g" % soft_accuray
print "Accuracy using CNN model %g" % cnn_accuray