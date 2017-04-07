#!/usr/bin/env python
# encoding: utf-8

import sys
import numpy
import input_data

def numpy_parzen(x, mu, sigma):
    a = ( x[:, None, :] - mu[None, :, :] ) / sigma
    
    def log_mean(i):
        return i.max(1) + numpy.log(numpy.exp(i - i.max(1)[:, None]).mean(1))
    
    return log_mean(-0.5 * (a**2).sum(2)) - mu.shape[1] * numpy.log(sigma * numpy.sqrt(numpy.pi * 2))


def main(sigma, sample_path='samples.npy'):
    
    # provide a .npy file where 10k generated samples are saved. 
    filename = sample_path
    
    print 'loading samples from %s'%filename
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    test_X, test_Y = mnist.test.next_batch(mnist.test.num_examples)
    samples = numpy.load(filename)
    
    test_ll = numpy_parzen(test_X, samples, sigma)

    print "Mean Log-Likelihood of test set = %.5f" % numpy.mean(test_ll)
    print "Std of Mean Log-Likelihood of test set = %.5f" % (numpy.std(test_ll) / 100)


if __name__ == "__main__":
    # Example: python likelihood_estimation_parzen.py 0.23 samples.npy
    main(float(sys.argv[1]), sys.argv[2])
    
