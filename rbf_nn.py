import numpy as np
from numpy.linalg import norm
from random import randint

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidPrime(x):
    return x*(1-x)

def model(x, layer, flag=0):
	centers = []
	for i in range(layer[0]):
		if(flag==0):
			c = np.random.randn(x.shape[1])
		else:
		    j = randint(0, x.shape[0]-1)
		    c = x[j]
		centers.append(c)

	sigmas = np.ones(layer[0])

	weights = []
	for i in range(1, len(layer)):
		w = np.random.randn(layer[i], layer[i-1])/ np.sqrt(layer[i-1])
		weights.append(w)
    
	bias = []
	for i in range(1, len(layer)):
		bias.append(np.zeros((1, layer[i])))
	return centers, sigmas, weights, bias

def feedforward(x, centers, sigmas, weights, bias):
    a_collection = [x]
    a = np.array([np.exp(-np.power(norm(x-centers[i], axis=1), 2)/(2*np.power(sigmas[i], 2))) for i in range(len(centers))]).T
    a_collection.append(a)
    for i in range(len(weights)):
        a = np.dot(a, weights[i].T) + bias[i]
        a_collection.append(a)
    return a_collection

def backpropagation(y, a_collection, centers, sigmas, weights, bias, learning_rate_c, learning_rate=1, l2_lambda=0):
    error = y - a_collection[-1]
    delta_collection = [error]
    
    for i in range(len(a_collection)-3, 0, -1):
        delta = np.dot(delta_collection[-1], weights[i])
        delta_collection.append(delta)
    
    delta_collection.append(np.dot(delta_collection[-1], weights[0]))
    
    delta_collection.reverse()
        
    cc = np.multiply(delta_collection[0], a_collection[1]).T
                
    for i in range(len(centers)):
        diff = a_collection[0] - centers[i]
        centers[i] += learning_rate_c*np.dot(cc[i], diff)/np.power(sigmas[i], 2)
        sigmas[i] += learning_rate*np.dot(cc[i], np.power(norm(diff, axis=1), 2))/np.power(sigmas[i], 3)
    
    for i in range(len(weights)):
        weights[i] += learning_rate*np.dot(delta_collection[i+1].T, a_collection[i+1]) + l2_lambda*weights[i]
        bias[i] += learning_rate*np.sum(delta_collection[i+1], axis=0)
    
    return centers, sigmas, weights, bias, delta_collection

def loss(y, pred):
    return np.mean(np.square(y - pred))