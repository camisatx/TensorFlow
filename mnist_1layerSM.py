import tensorflow as tf
import time

import input_data

""" Train a model to look at images and predict what digits they are. Uses
MNIST data from Yann LeCun (http://yann.lecun.com/exdb/mnist/).

http://tensorflow.org/tutorials/mnist/pros/index.md
"""

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# A placeholder that will be input when the computation is run
x = tf.placeholder('float', [None, 784])    # 'None' means any length
# Placeholder to input the correct answers, indicated by the digit (0-9)
y_ = tf.placeholder('float', [None, 10])

# Weights for model that will be modified during computation
W = tf.Variable(tf.zeros([784, 10]))
# Biases for model that will be modified during computation
b = tf.Variable(tf.zeros([10]))

# Implement the softmax model
# Multiply vectorized input images by the weight matrix and then add the bias
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Use cross-entropy as the cost function to be minimized
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# Minimize the cross-entropy function using gradient descent (0.01 step length)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Initialize the variables
init = tf.initialize_all_variables()

# Launch the model in a Session
with tf.Session() as sess:

    # Run the variable initialization operation
    sess.run(init)

    time_1 = time.time()
    # Train the model 1000 times, with each loop using 100 random points from
    # set. Using small batches of random data is called stochastic training.
    for i in range(1000):
        # Batch replaces the placeholder; it's an n by 784 array
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    # Find accuracy of model: y is the test output and y_ is the actual label
    # Returns a list of booleans
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # Change the list of booleans to floats, then takes the mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    print('One layer SoftMax network MNIST training took %s seconds with an '
          'accuracy of %g' %
          (time.time() - time_1,
           accuracy.eval(feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels})))
