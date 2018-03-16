import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidPrime(x):
    return x*(1-x)

def model(layer):
    weights = []
    for i in range(1, len(layer)):
        w = np.random.randn(layer[i], layer[i-1])/ np.sqrt(layer[i-1])
        weights.append(w)
    
    bias = []
    for i in range(1, len(layer)):
        bias.append(np.zeros((1, layer[i])))
    return weights, bias

def feedforward(x, weights, bias):
    a_collection = [x]
    a = x
    for i in range(len(weights)):
        z = np.dot(a, weights[i].T) + bias[i]
        a = sigmoid(z)
        a_collection.append(a)
    return a_collection

def backpropagation(y, vw_old_collection, vb_old_collection, a_collection, weights, bias, alpha=1, learning_rate=1, l2_lambda=0):
    error = y - a_collection[-1]
    delta_collection = [error*sigmoidPrime(a_collection[-1])]
    
    for i in range(len(a_collection)-2, 0, -1):
        delta = np.dot(delta_collection[-1], weights[i])*sigmoidPrime(a_collection[i])
        delta_collection.append(delta)
    
    delta_collection.reverse()

    vw_collection = []
    vb_collection = []
    for i in range(len(weights)):
        gradw = np.dot(delta_collection[i].T, a_collection[i])
        gradb = np.sum(delta_collection[i], axis=0)
        vw = alpha*vw_old_collection[i] + learning_rate*gradw
        vb = alpha*vb_old_collection[i] + learning_rate*gradb
        vw_collection.append(vw)
        vb_collection.append(vb)

        weights[i] += vw + l2_lambda*weights[i]
        bias[i] += vb
    return weights, bias, vw_collection, vb_collection

def loss(y, pred):
    return np.mean(np.square(y - pred))