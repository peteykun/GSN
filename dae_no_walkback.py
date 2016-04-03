#!env/bin/python
import tensorflow as tf
import input_data
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy

# Helpers
trunc = lambda x : str(x)[:8]

def binomial_draw(shape=[1], p=0.5, dtype='float32'):
  return tf.select(tf.less(tf.random_uniform(shape=shape, minval=0, maxval=1, dtype='float32'), tf.fill(shape, p)), tf.ones(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))

def salt_and_pepper(X, rate=0.3):
  a = binomial_draw(shape=tf.shape(X), p=1-rate, dtype='float32')
  b = binomial_draw(shape=tf.shape(X), p=0.5, dtype='float32')
  z = tf.zeros(tf.shape(X), dtype='float32')
  c = tf.select(tf.equal(a, z), b, z)
  return tf.add(tf.mul(X, a), c)

# Read mnist examples
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Number of hidden layers
hidden_size = 2000

# Number of epochs
n_epoch = 200

# Batch size
batch_size = 100

# Salt and pepper noise
input_salt_and_pepper = 0.4

# Input
x0 = tf.placeholder(tf.float32, [None, 784])

W1 = tf.Variable(tf.random_normal([784, hidden_size]))
b0 = tf.Variable(tf.random_normal([784]))
b1 = tf.Variable(tf.random_normal([hidden_size]))

# Add noise
x_corrupt = salt_and_pepper(x0, input_salt_and_pepper)
# Activate
h1 = tf.sigmoid(tf.matmul(x_corrupt, W1) + b1)
# Activate
x1 = tf.sigmoid(tf.matmul(h1, tf.transpose(W1)) + b0)

cross_entropy = -tf.reduce_sum(x0*tf.log(tf.clip_by_value(x1,1e-10,1.0)) + (1-x0)*tf.log(tf.clip_by_value(1-x1,1e-10,1.0)))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

# Initalization
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(n_epoch):
  print 'Epoch: ', i+1,

  # train
  train_cost = []
  for j in range(mnist.train.num_examples/batch_size):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    result = sess.run((cross_entropy, train_step), feed_dict={x0: batch_xs})
    train_cost.append(result[0])

  train_cost = numpy.mean(train_cost)
  print 'Train: ', trunc(train_cost),

  # valid
  valid_cost = []
  for j in range(mnist.validation.num_examples/batch_size):
    batch_xs, batch_ys = mnist.validation.next_batch(batch_size)
    result = sess.run(cross_entropy, feed_dict={x0: batch_xs})
    valid_cost.append(result)

  valid_cost = numpy.mean(valid_cost)
  print 'Valid: ', trunc(valid_cost),

  # test
  test_cost = []
  for j in range(mnist.test.num_examples/batch_size):
    batch_xs, batch_ys = mnist.test.next_batch(batch_size)
    result = sess.run(cross_entropy, feed_dict={x0: batch_xs})
    test_cost.append(result)

  test_cost = numpy.mean(test_cost)
  print 'Test: ', trunc(test_cost)

# sample from the network
test_input = mnist.test.next_batch(1)[0]
samples = [test_input]

fig, axs = plt.subplots(40, 10, figsize=(10, 40))

for i in range(400):
  samples.append(sess.run(x1, feed_dict={x0: samples[-1]}))

  axs[i/10][i%10].imshow(numpy.reshape(samples[i], (28,28)), cmap='gray')

plt.axis('off')
plt.savefig('dae_no_walkback.png')
