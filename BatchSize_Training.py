# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 21:25:54 2017

@author: Razor
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("C:/tmp/batch_training/data/", one_hot=True)

# Training Parameters
learning_rate = 0.0001
num_steps = 200
batch_size = 32
display_step = 10
logs_path = "C:/tmp/batch_training/tensorflow_logs"

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, [None, num_input], name='InputData')
    Y_theo = tf.placeholder(tf.float32, [None, num_classes], name='LabelData')

keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Store layers weights
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name='conv_weights_1'),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name='conv_weights_2'),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024]), name='fully_weights_1'),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]), name='softmax_weights')
}
# Store layer biases
biases = {
    'bc1': tf.Variable(tf.random_normal([32]), name='conv_biases_1'),
    'bc2': tf.Variable(tf.random_normal([64]), name='conv_biases_2'),
    'bd1': tf.Variable(tf.random_normal([1024]), name='fully_biases_1'),
    'out': tf.Variable(tf.random_normal([num_classes]), name='softmax_biases_1')
}

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out']) 
    return out

# Construct model
with tf.name_scope('Model'):
    logits = conv_net(X, weights, biases, keep_prob)
    Y_pract = tf.nn.softmax(logits)

# Define loss and optimizer
with tf.name_scope('Loss'):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y_theo))
    # Create a summary to monitor the loss 
    #tf.summary.scalar('loss_op', loss_op)

# Define optimizer
with tf.name_scope('AdamOptimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

# Evaluate model
with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(Y_pract, 1), tf.argmax(Y_theo, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # Create a summary to monitor accuracy
    #tf.summary.scalar('Accuracy', accuracy)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
merged_summary = tf.summary.merge_all()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)  
    # Write all summaries to TensorBoard
    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
    # Merge all summaries into a single op
    for step in range(1, num_steps+1):  
        batch_x, batch_y = mnist.train.next_batch(batch_size)        
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, 
                                      Y_theo: batch_y, 
                                      keep_prob: 0.5})       
        if step % display_step == 0 or step == 1:            
            # Calculate batch loss and accuracy
            training_accuracy, loss, summary = sess.run(
                    [accuracy, loss_op, merged_summary], feed_dict={X: batch_x, 
                                                    Y_theo: batch_y, 
                                                    keep_prob: 1.0})
            summary_writer.add_summary(summary, step)          
            print("Step " + str(step) + ", Loss = " + "{:.3f}".format(loss) + \
                  ", Training Accuracy = " + "{:.3f}".format(training_accuracy))
    
    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y_theo: mnist.test.labels[:256],
                                      keep_prob: 1.0}))