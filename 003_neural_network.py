"""
Neural Network from scratch

This code is still pending changes, unless this comment line is removed.
"""

import random
import numpy as np

def single_neuron_manual():

    x = [0.2,2,3]
    w = [-0.35,0.5,0.9]
    b = -0.23
    res = 0

    for i in range(3):
        res += x[i]*w[i]

    res += b
    res = 1 / (1 + 2.718 ** -res)

    print(res)

def single_neuron_numpy():

    x = [0.2,2,3]
    w = [-0.35,0.5,0.9]
    b = -0.23
    res = np.dot(w,x)
    res += b
    res = 1 / (1 + 2.718 ** -res)

    print(res)

single_neuron_manual()
single_neuron_numpy()