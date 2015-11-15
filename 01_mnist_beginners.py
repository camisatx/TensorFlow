import input_data
import tensorflow as tf

"""
Train a model to look at images and predict what digits they are. Uses MNIST
data from Yann LeCun (http://yann.lecun.com/exdb/mnist/).

http://tensorflow.org/tutorials/mnist/beginners/index.md
"""

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# A placeholder that will be input when the computation is run
x = tf.placeholder('float', [None, 784])    # 'None' means any length

# Weights for model
W = tf.Variable(tf.zeros([784, 10]))
# Biases for model
b = tf.Variable(tf.zeros([10]))

# Implement the softmax model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Placeholder to input the correct answers
y_ = tf.placeholder('float', [None, 10])

# Use cross-entropy as the cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# Minimuze the cross-entropy function using gradient descent
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Initialize the variables
init = tf.initialize_all_variables()

# Launch the model in a Session, running the variable initialization operation
sess = tf.Session()
sess.run(init)

# Train the model 1000 times, with each loop using 100 random points from set
# Using small batches of random data is called stochastic training
for i in range(1000):
    # batch data replaces the placeholder data
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Find accuracy of model: y is the test output and y_ is the actual label
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
