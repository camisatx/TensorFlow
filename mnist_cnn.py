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


def weight_variable(shape):
    """ Weights for model that will be modified during computation. Weights
    contain small amount of noise for symmetry breaking and to prevent 0
    gradients. Since CNN is using ReLU neurons, it's good practice to initialize
    them with a slightly positive initial bias to avoid 'dead neurons'.

    :param shape: List of array dimension
    """

    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """ Biases for model that will be modified during computation. Since CNN
    is using ReLU neurons, it's good practice to initialize them with a
    slightly positive initial bias to avoid 'dead neurons'.

    :param shape: List of array dimension
    """

    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """ Convolutions uses stride of one and are zero padded so that the output
    is the same size as the input.

    :param x: Image array
    :param W: Weight array
    """

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """ Use max pooling over 2x2 blocks

    :param x: Image array
    """

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

################################################################################
# First convolutions layer that will compute 32 features for each 5x5 patch.

# First two dimensions are the patch size, the next is the number of input
#   channels and the last is the number of output channels.
W_conv1 = weight_variable([5, 5, 1, 32])
# Bias vector with a component for each output channel
b_conv1 = bias_variable([32])

# Apply the layer, reshaping x to a 4d tensor, with the second and third
#   dimension corresponding to image width and height, and the final dimension
#   corresponding to the number of color channels.
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolve x_image with the weight tensor, add the bias, apply the ReLu
#   function and max pool function.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_poll1 = max_pool_2x2(h_conv1)

################################################################################
# Second convolution layer that will compute 64 features for each 5x5 patch.

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_poll1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

################################################################################
# Fully-connected layer with 1024 neurons; allows processing on the entire image

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

################################################################################
# Apply dropout before the readout layer to reduce over fitting

# Placeholder allows system to turn dropout on during training, and turn it
#   off during testing
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

################################################################################
# Add softmax layer

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

################################################################################


# Launch the model in a Session
with tf.Session() as sess:

    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimzer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    sess.run(tf.initialize_all_variables())

    time_1 = time.time()

    for i in range(20000):
        batch = mnist.train.next_batch(50)

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0],
                                                      y_: batch[1],
                                                      keep_prob: 1.0})
            print('Step %d: training accuracy %g' % (i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('Convolution Neural Net MNIST training took %s seconds with an '
          'accuracy of %g' %
          (time.time() - time_1, accuracy.eval(feed_dict={x: mnist.test.images,
                                                          y_: mnist.test.labels,
                                                          keep_prob: 1.0})))
