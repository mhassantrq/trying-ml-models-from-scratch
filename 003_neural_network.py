"""
Neural Network from scratch

This code is still pending changes, unless this comment line is removed.
"""

import random

def single_neuron():

    x = [random.random() for _ in range(3)]
    w = [random.random() for _ in range(3)]

    b = -0.23
    res = 0

    for i in range(3):
        res += x[i]*w[i]

    res += b
    res = 1 / (1 + 2.718 ** -res)

    print(res)


single_neuron()