#!env/bin/python
import tensorflow as tf
import input_data
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy

# Helpers
trunc = lambda x : str(x)[:8]
cast32 = lambda x : numpy.cast['float32'](x)

def binomial_draw(shape=[1], p=0.5, dtype='float32'):
  return tf.select(tf.less(tf.random_uniform(shape=shape, minval=0, maxval=1, dtype='float32'), tf.fill(shape, p)), tf.ones(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))

def binomial_draw_vec(p_vec, dtype='float32'):
  shape = tf.shape(p_vec)
  return tf.select(tf.less(tf.random_uniform(shape=shape, minval=0, maxval=1, dtype='float32'), p_vec), tf.ones(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))

def salt_and_pepper(X, rate=0.3):
  a = binomial_draw(shape=tf.shape(X), p=1-rate)
  b = binomial_draw(shape=tf.shape(X), p=0.5)
  z = tf.zeros(tf.shape(X), dtype='float32')
  c = tf.select(tf.equal(a, z), b, z)
  return tf.add(tf.mul(X, a), c)

def add_gaussian_noise(X, sigma):
  noise = tf.random_normal(tf.shape(X), stddev=sigma, dtype=tf.float32)
  return tf.add(X, noise)

# Xavier Initializers
def get_shared_weights(n_in, n_out, interval):
    val = numpy.random.uniform(-interval, interval, size=(n_in, n_out))
    val = cast32(val)
    return tf.Variable(val)

def get_shared_bias(n, offset = 0):
    val = numpy.zeros(n) - offset
    val = cast32(val)
    return tf.Variable(val)

# Read mnist examples
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Number of hidden layers
hidden_size = 1500

# Number of walkbacks
walkbacks = 4

# Number of epochs
n_epoch = 1000

# Batch size
batch_size = 100

# Salt and pepper noise
input_salt_and_pepper = 0.4

# Noisy neurons
hidden_add_noise_sigma = 2

# Input
x0 = tf.placeholder(tf.float32, [None, 784])
x0_copy = x0

W1 = get_shared_weights(784, hidden_size, numpy.sqrt(6. / (784 + hidden_size)))
W2 = get_shared_weights(hidden_size, hidden_size, numpy.sqrt(6. / (hidden_size + hidden_size)))
b0 = get_shared_bias(784)
b1 = get_shared_bias(hidden_size)
b2  = get_shared_bias(hidden_size)

p_X_chain = []
h2 = tf.zeros([1, hidden_size])

for i in range(walkbacks):
  # Add noise
  x_corrupt = salt_and_pepper(x0_copy, input_salt_and_pepper)
  # Activate
  h1 = tf.tanh(tf.matmul(x_corrupt, W1) + tf.matmul(h2, tf.transpose(W2)) + b1)
  # Activate
  h2 = add_gaussian_noise(tf.tanh(add_gaussian_noise(tf.matmul(h1, W2) + b2, hidden_add_noise_sigma)), hidden_add_noise_sigma)
  # Activate
  x1 = tf.sigmoid(tf.matmul(h1, tf.transpose(W1)) + b0)
  # Build the reconstruction chain
  p_X_chain.append(x1)
  # Input sampling
  x0_copy = binomial_draw_vec(x1)

cross_entropies = [-tf.reduce_sum(x0*tf.log(tf.clip_by_value(x1,1e-10,1.0)) + (1-x0)*tf.log(tf.clip_by_value(1-x1,1e-10,1.0))) for x1 in p_X_chain]
cross_entropy = tf.add_n(cross_entropies)
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
  print 'Train: ', train_cost/(100*28*28),

  # valid
  valid_cost = []
  for j in range(mnist.validation.num_examples/batch_size):
    batch_xs, batch_ys = mnist.validation.next_batch(batch_size)
    result = sess.run(cross_entropy, feed_dict={x0: batch_xs})
    valid_cost.append(result)

  valid_cost = numpy.mean(valid_cost)
  print 'Valid: ', valid_cost/(100*28*28),

  # test
  test_cost = []
  for j in range(mnist.test.num_examples/batch_size):
    batch_xs, batch_ys = mnist.test.next_batch(batch_size)
    result = sess.run(cross_entropy, feed_dict={x0: batch_xs})
    test_cost.append(result)

  test_cost = numpy.mean(test_cost)
  print 'Test: ', test_cost/(100*28*28)

  if i%50 == 0:
    # sample from the network
    test_input = mnist.test.next_batch(1)[0]
    samples = [test_input]

    fig, axs = plt.subplots(40, 10, figsize=(10, 40))

    for j in range(10000):
      samples.append(sess.run(x1, feed_dict={x0: samples[-1]}))
      if j < 400:
        axs[j/10][j%10].imshow(numpy.reshape(samples[j], (28,28)), cmap='gray')

    numpy.save('gsn' + str(i) + '.npy', numpy.array(samples[1:10001,:,:]))
    plt.axis('off')
    plt.savefig('gsn' + str(i) + '.png')
