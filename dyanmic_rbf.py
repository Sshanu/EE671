import numpy as np
from rbf_nn import sigmoid, sigmoidPrime, feedforward, backpropagation, loss, model
import matplotlib.pyplot as plt
from random import randint
import copy
from numpy.linalg import norm


p = np.zeros((2, 2000))
u = []
t = []
d = [0.5]
for i in range(2000):
    u.append(2*np.random.uniform(0, 1)-1)
    dd = d[-1]/(1+d[-1]*d[-1]) + np.power(u[-1], 3)
    d.append(dd)
    p[0][i] = u[i]
    p[1][i] = d[i]
    t.append(dd)
    
t = np.array(t)/3 + 0.5
plt.plot(u, t, marker='o', markersize=2, linewidth=0, color='r')
plt.show()


X = np.array(p).T
y = np.array(t)
y = np.reshape(y, (np.size(y), 1))




layer = [50, 20, 1]
learning_rate = 0.001
learning_rate_c = 0.005
l2_lambda = 0
batch_size = 100
num_batch = int(X.shape[0]/ batch_size)
input_dim = X.shape[1]

np.random.seed(0)
init_center = copy.deepcopy(X)
centers, sigmas, weights, bias = model(init_center, layer)

for i in range(5000):
    for rr in range(num_batch):
        input_x = X[batch_size*rr:batch_size*(rr+1)]
        input_y = y[batch_size*rr:batch_size*(rr+1)]

        aa = feedforward(input_x[:], centers, sigmas, weights, bias)
        centers, sigmas, weights, bias, dd = backpropagation(input_y[:], aa, centers, sigmas, weights, bias, learning_rate_c, learning_rate=learning_rate, l2_lambda=l2_lambda)

    if(i%100==0):
        aa = feedforward(X[:], centers, sigmas, weights, bias)
        yd = aa[-1]
        lo = loss(y, yd)
        print(i, lo)




# Prediction
aa = feedforward(X, centers, sigmas, weights, bias)
yd = aa[-1]

fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(u,(yd-0.5)*3, marker='o', markersize=5, linewidth=0, color='orange')
plt.plot(u,(t-0.5)*3, marker='^', markersize=5, linewidth=0, color='g')
plt.show()