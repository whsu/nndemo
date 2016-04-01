import numpy as np

def squared_loss(A, Y):
	d = A-Y
	return 0.5*np.sum(d*d)/len(Y)

def squared_loss_dv(A, Y):
	return A-Y

class Loss:
	SQUARED = "squared"

LOSS = {
	Loss.SQUARED:squared_loss,
}

LOSSDV = {
	Loss.SQUARED:squared_loss_dv,
}


