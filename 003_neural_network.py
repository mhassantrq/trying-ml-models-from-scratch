"""
Neural Network from scratch

This code is still pending changes, unless this comment line is removed.
"""

import random
import numpy as np

class NeuralNetwork:
    pass

class Neuron:
    def single_neuron_manual(self):

        x = [0.2,2,3]
        w = [-0.35,0.5,0.9]
        b = -0.23
        res = 0

        for i in range(3):
            res += x[i]*w[i]

        res += b
        res = 1 / (1 + 2.718 ** -res)

        print(res)

    def single_neuron_numpy(self):

        x = [0.2,2,3]
        w = [-0.35,0.5,0.9]
        b = -0.23
        res = np.dot(w,x)
        res += b
        res = 1 / (1 + 2.718 ** -res)

        print(res)

    def single_layer(self):
        X = [0.35, -0.6, 1.2]
        w = [[0.2, 1.1, 0.7],
             [1.9, 0.65, -0.53]]
        b = [0.9, 0.85]
        outputs = np.dot(w, X) + b
        print(outputs)


n = Neuron()
n.single_layer()
