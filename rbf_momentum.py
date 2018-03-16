import numpy as np
from numpy.linalg import norm
from random import randint

def rbf(x, centers, sigmas):
	return np.array([np.exp(-np.power(norm(x- centers[i], axis=1), 2)/(2*np.power(sigmas[i], 2)))  for i in range(len(centers))])

def model(x, layer, flag=0):
	centers = []
	for i in range(layer[0]):
		if(flag==0):
			c = np.random.randn(x.shape[1])
		else:
			c = x[randint(0, x.shape[0]-1)]
		centers.append(c)
	sigmas = np.ones(layer[0])

	weights = []
	bias = []
	for i in range(1, len(layer)):
		w = np.random.randn(layer[i], layer[i-1])/ np.sqrt(layer[i-1])
		weights.append(w)
		bias.append(np.zeros((1, layer[i])))
	return centers, sigmas, weights, bias

def feedforward(x, centers, sigmas, weights, bias):
	a = x
	a_collection = [a]
	a = rbf(x, centers, sigmas).T
	a_collection.append(a)
	for i in range(len(weights)):
		a = np.dot(a_collection[-1], weights[i].T) + bias[i]
		a_collection.append(a)
	return a_collection

def backpropagation(y, vc_old_collection,  vs_old_collection, vw_old_collection, vb_old_collection, a_collection, centers, sigmas, weights, bias, alpha_c, alpha, lr, lr_c):
	error = y - a_collection[-1]
	delta_connection = [error]
	for i in range(len(a_collection)-3, -1, -1):
		delta = np.dot(delta_connection[-1], weights[i])
		delta_connection.append(delta)

	delta_connection.reverse()

	temp = (delta_connection[0]*a_collection[1]).T

	vw_collection = []
	vb_collection = []
	vs_collection = []
	vc_collection = []
	for i in range(len(centers)):
		diff = a_collection[0] - centers[i]
		gradc = np.dot(temp[i], diff)/np.power(sigmas[i], 2)
		grads = np.dot(temp[i], np.power(norm(diff, axis=1), 2))/ np.power(sigmas[i], 3)

		vc = alpha_c*vc_old_collection[i] + lr_c*gradc
		vs = alpha_c*vs_old_collection[i] + lr_c*grads

		vc_collection.append(vc)
		vs_collection.append(vs)

		centers[i] += vc
		sigmas[i] += vs

	for i in range(len(weights)):
		gradw = np.dot(delta_connection[i+1].T, a_collection[i+1])
		gradb = np.sum(delta_connection[i+1], axis=0)

		vw = alpha*vw_old_collection[i] + lr*gradw
		vb = alpha*vb_old_collection[i] + lr*gradb

		vw_collection.append(vw)
		vb_collection.append(vb)
		weights[i] += vw
		bias[i] += vb

	return centers, sigmas, weights, bias, vc_collection, vs_collection, vw_collection, vb_collection


def loss(y, pred):
	return np.mean(np.square(y - pred))