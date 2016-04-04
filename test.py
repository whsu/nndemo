import unittest

import numpy as np

from neural_network import *
from activation_functions import *
from loss_functions import *

def onehot(Y, K):
	N = len(Y)
	Y_new = np.zeros((N, K), dtype=int)
	for n in range(N):
		Y_new[n,Y[n]] = 1
	return Y_new

class TestNeuralNetwork(unittest.TestCase):

	def test1(self):
		print('\n\n***** Testing classification *****\n')
		np.random.seed(0)
		clf = NeuralNetwork(numnodes=[1,10,2],
		                    activations=[Activation.SIGMOID, Activation.SIGMOID],
		                    loss=Loss.SQUARED,
		                    alpha=0.001,
		                    maxiter=1000,
		                    batchsize=256,
		                    momentum=0.5,
		                    printfreq=100)
		X = np.array([[-4.0],[-3.0],[-2.0],[-1.0],[1.0],[2.0],[3.0],[4.0]])
		y = np.array([0,0,0,0,1,1,1,1])
		clf.fit(X, onehot(y, 2))
		c = clf.classify(X)
		print(c)
		np.testing.assert_array_equal(c, y)

	def test2(self):
		print('\n\n***** Testing binary classification *****\n')
		np.random.seed(0)
		clf = NeuralNetwork(numnodes=[1,10,1],
		                    activations=[Activation.SIGMOID, Activation.SIGMOID],
		                    loss=Loss.SQUARED,
		                    alpha=0.001,
		                    maxiter=1000,
		                    batchsize=256,
		                    momentum=0.5,
		                    printfreq=100)
		X = np.array([[-4.0],[-3.0],[-2.0],[-1.0],[1.0],[2.0],[3.0],[4.0]])
		y = np.array([0,0,0,0,1,1,1,1])
		clf.fit(X, y.reshape(8,1))
		c = clf.classify_bin(X)
		print(c)
		np.testing.assert_array_equal(c, y)

	def test3(self):
		print('\n\n***** Testing regression tanh *****\n')
		np.random.seed(0)
		clf = NeuralNetwork(numnodes=[1,10,1],
		                    activations=[Activation.TANH, Activation.LINEAR],
		                    loss=Loss.SQUARED,
		                    alpha=0.001,
		                    maxiter=100000,
		                    batchsize=256,
		                    momentum=0.5,
		                    printfreq=10000)
		X = np.array([[-4.0],[-3.0],[-2.0],[-1.0],[1.0],[2.0],[3.0],[4.0]])
		y = np.array([15.9,9.2,3.85,1.13,1.07,4.1,8.89,16.2]).reshape(8,1)
		clf.fit(X, y)
		c = clf.predict(X)
		print(c)
		np.testing.assert_array_almost_equal(c, y, decimal=1)

	def test4(self):
		print('\n\n***** Testing regression relu *****\n')
		np.random.seed(0)
		clf = NeuralNetwork(numnodes=[1,10,10,1],
		                    activations=[Activation.RELU, Activation.RELU, Activation.LINEAR],
		                    loss=Loss.SQUARED,
		                    alpha=0.001,
		                    maxiter=100000,
		                    batchsize=256,
		                    momentum=0.5,
		                    printfreq=10000)
		X = np.array([[-4.0],[-3.0],[-2.0],[-1.0],[1.0],[2.0],[3.0],[4.0]])
		y = np.array([15.9,9.2,3.85,1.13,1.07,4.1,8.89,16.2]).reshape(8,1)
		clf.fit(X, y)
		c = clf.predict(X)
		print(c)
		np.testing.assert_array_almost_equal(c, y, decimal=6)

if __name__ == '__main__':
    unittest.main()

