'''An implementation of fully connected feedforward neural network'''

import numpy as np

from activation_functions import *
from loss_functions import *

class NeuralNetwork:
    '''A fully connected feedforward neural network.

       Parameters
       ----------
       numnodes : list of int
           Number of nodes in each layer, from input to output.
       activations: list of {'linear', 'relu', 'sigmoid', 'tanh'}
           Name of activation function in each layer, from first hidden
           layer to output layer.
       loss: {'squared'}
           Name of loss function. 
       alpha: float
           Gradient descent step size.
       maxiter: int
           Maximum number of iterations for gradient descent.
       batchsize: int
           Gradient descent batch size.
       momentum: float
           Gradient descent momentum.
       printfreq: int
           Print loss every printfreq iteration.
           Set to 0 to disable printing.
    '''

    def __init__(self, numnodes, activations, loss, alpha, maxiter,
                       batchsize, momentum, printfreq):
        assert len(numnodes)==len(activations)+1
        self.numnodes = numnodes
        self.activations = [None] + activations
        self.loss = loss
        self.L = len(activations)
        self.alpha = alpha
        self.maxiter = maxiter
        self.batchsize = batchsize
        self.momentum = momentum
        self.printfreq = printfreq

        self._init_weights()
        self._init_velocities()

    def _init_weights(self):
        self.w = [None] * (self.L+1)
        for layer in range(1,self.L+1):
            N = self.numnodes[layer]
            M = self.numnodes[layer-1]
            self.w[layer] = np.empty((N, M+1))
            self.w[layer][:,0] = 0.25
            self.w[layer][:,1:] = np.random.rand(N, M) - 0.5

            ind = np.arange(N)
            for i in range(1, M+1):
                np.random.shuffle(ind)
                self.w[layer][ind[:N/2],i] = 0.0

    def _init_velocities(self):
        self.v = [None] * (self.L+1)
        for i in range(1, self.L+1):
            self.v[i] = np.zeros_like(self.w[i])

    def forwardprop(self, X):
        z = [None] * (self.L+1)
        a = [None] * (self.L+1)
        a[0] = X
        for i in range(1,self.L+1):
            z[i] = self.w[i][:,1:].dot(a[i-1].T).T + self.w[i][:,0]
            a[i] = ACT[self.activations[i]](z[i])
        return z, a

    def backprop(self, X, Y):
        D = [None] * (self.L+1)
        dw = [None] * (self.L+1)

        z, a = self.forwardprop(X)

        for i in range(self.L,0,-1):
            if i == self.L:
                D[i] = ACTDV[self.activations[i]](z[i]) * self.compute_loss_dv(a[i], Y)
            else:
                D[i] = ACTDV[self.activations[i]](z[i]) * D[i+1].dot(self.w[i+1][:,1:])
            dw[i] = np.empty_like(self.w[i])
            dw[i][:,0] = np.sum(D[i], axis=0)
            dw[i][:,1:] = D[i].T.dot(a[i-1])

        return dw

    def fit(self, X, Y):
        '''Build a neural network from the training set (X, Y).'''
        (N, D) = X.shape
        assert D == self.numnodes[0]

        indices = np.arange(N)

        loss_old = self.compute_loss(X, Y)
        if self.printfreq > 0:
            print('iter={0}, loss={1}'.format(0, loss_old))

        for i in range(1,self.maxiter+1):
            np.random.shuffle(indices)
            for n in range(0,N,self.batchsize):
                ind = indices[n*self.batchsize:(n+1)*self.batchsize]
                dw = self.backprop(X[ind], Y[ind])
                for layer in range(1,self.L+1):
                    self.v[layer] = self.momentum * self.v[layer] + self.alpha * dw[layer]
                    self.w[layer] -= self.v[layer]

            loss_new = self.compute_loss(X, Y)
            if loss_new > loss_old:
                self.alpha = 1/(1.0/self.alpha+1)
            loss_old = loss_new
            if self.printfreq > 0 and i % self.printfreq == 0:
                print('iter={0}, loss={1}'.format(i, loss_new))

    def compute_loss(self, X, Y):
        ao = self.predict(X)
        return LOSS[self.loss](ao, Y)

    def compute_loss_dv(self, a, Y):
        return LOSSDV[self.loss](a, Y)

    def predict(self, X):
        '''Predict values at output layer for X.'''
        z, a = self.forwardprop(X)
        return a[-1]

    def classify_prob(self, X):
        ao = self.predict(X)
        return (ao.T / np.sum(ao,axis=1)).T

    def classify(self, X):
        p = self.classify_prob(X)
        return np.argmax(p, axis=1)

    def classify_bin_prob(self, X):
        ao = self.predict(X)
        return ao.flatten()

    def classify_bin(self, X, pthresh=0.5):
        p = self.classify_bin_prob(X)
        return (p>pthresh).astype(int)

