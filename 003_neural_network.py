"""
Neural Network from scratch

This code is still pending changes, unless this comment line is removed.
"""

import random
import numpy as np
import math

class NeuralNetwork:
    def three_layer(self):
        X = [random.random() for _ in range(3)]

        w1 = [[random.random() for _ in range(3)],
            [random.random() for _ in range(3)]]
        b1 = [random.random() for _ in range(2)]

        w2 = [[random.random() for _ in range(2)],
            [random.random() for _ in range(2)]]
        b2 = [random.random() for _ in range(2)]

        w3 = [random.random() for _ in range(2)]
        b3 = [random.random()]

        output1 = np.dot(X, np.array(w1).T) + b1
        output2 = np.dot(output1, np.array(w2).T) + b2
        output3 = np.dot(output2, np.array(w3).T) + b3

        print(output1)
        print(output2)
        print(output3)
        sig = 1/(1+math.exp(-output3[0]))
        print(f'Sigmoid: {sig}')

n = NeuralNetwork()
n.three_layer()