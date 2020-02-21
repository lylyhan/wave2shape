import torch
from kymatio import Scattering1D
from torch.autograd import backward

def getsc(y,j):
	"""
	this function outputs scattering transform of a time-domain signal.
	"""
	N = 2**j
	scattering = Scattering1D(J = 10,shape=(N,))
	Sy = scattering(y)
	return Sy


def regress_signal(s_c, j, learning_rate=0.1, n_iterations=100):
	"""
	this function finds time-domain signal from scattering transform
	input: scattering tranform tensor, length of the signal(implicitly), learning rate and number of iterations
	output: time-domain signal represented in a tensor
	"""

	N = 2**j
	scattering = Scattering1D(J = 10,shape=(N,))

	#random guess
	x = torch.randn((N,),requires_grad=True)
	Sx = scattering(x)
	#target
	Sy = s_c

	#x.requires_grad = True
	#n_iterations = 100
	#iterate to regress random guess to be close to target
	for it_id in range(100):
	    # Backpropagation
	    backward(torch.norm(Sx - Sy))
	    delta_x = x.grad
	    print(x.grad)

	    # Gradient descent
	    with torch.no_grad():
	        x = x - delta_x * learning_rate
	    x.requires_grad = True

	    # New forward propagation
	    Sx = scattering(x)
	    # Measure the new loss
	    print(torch.norm(Sx - Sy))


	
	return x


