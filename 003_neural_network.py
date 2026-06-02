"""
Neural Network from scratch

This code is still pending changes, unless this comment line is removed.
"""

import random
import numpy as np

class NeuralNetwork:
    def three_layer(self):
        X = [[random.random() for _ in range(3)],
            [random.random() for _ in range(3)],
            [random.random() for _ in range(3)]]

        w1 = [[random.random() for _ in range(3)],
            [random.random() for _ in range(3)]]
        b1 = [random.random() for _ in range(2)]

        w2 = [[random.random() for _ in range(3)],
            [random.random() for _ in range(3)]]
        b2 = [random.random() for _ in range(2)]

        output1 = np.dot(X, np.array(w1).T) + b1
        output1 = np.dot(X, np.array(w1).T) + b1

#        print(outputs)
#   pending...


n = NeuralNetwork()
n.double_layer()