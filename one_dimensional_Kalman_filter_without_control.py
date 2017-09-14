# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:41:29 2017

@author: L.J.
"""
import numpy as np
from matplotlib import pyplot as plt

N = 100
t = np.array(range(N))
z = 100/(np.exp(-(t-N/2)/N*20) + 1)

w = 5 * np.random.randn(N) # sensor noise
z = z + w

def predict(x,p):
    q = 1 # process noise
    x = x
    p = p + q
    return x, p

def update(x, p, z):
    r = 10 # sensor noise
    k = p / (p + r)
    x = x + k*(z - x)
    p = p - k * p
    return x, p

x = np.zeros(N)
x[0] = z[0]
for i in range(1, N):
    (x[i], p) = predict(x[i-1], 10)
    (x[i], p) = update(x[i], p, z[i])
    
plt.plot(t, z, 'r', label = 'raw')
plt.plot(t, x, 'b', label = 'filtered')
plt.legend(loc='upper left')
plt.show()

