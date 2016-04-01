import numpy as np

def relu_activation(z):
	return np.maximum(0.0, z)

def relu_activation_dv(z):
	return (z>0).astype(float)

def sigmoid_activation(z):
	return 1/(1+np.exp(-z))

def sigmoid_activation_dv(z):
	s = sigmoid_activation(z)
	return s*(1-s)

def tanh_activation(z):
	return np.tanh(z)

def tanh_activation_dv(z):
	t = tanh_activation(z)
	return 1-t*t

def linear_activation(z):
	return z

def linear_activation_dv(z):
	return 1.0

class Activation:
	RELU = 'relu'
	SIGMOID = 'sigmoid'
	TANH = 'tanh'
	LINEAR = 'linear'

ACT = {
	Activation.RELU:relu_activation,
	Activation.SIGMOID:sigmoid_activation,
	Activation.TANH:tanh_activation,
	Activation.LINEAR:linear_activation,
}

ACTDV = {
	Activation.RELU:relu_activation_dv,
	Activation.SIGMOID:sigmoid_activation_dv,
	Activation.TANH:tanh_activation_dv,
	Activation.LINEAR:linear_activation_dv,
}


