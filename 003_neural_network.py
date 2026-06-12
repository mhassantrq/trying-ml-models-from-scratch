"""
Neural Network from scratch

This code is still pending changes, unless this comment line is removed.
"""

import random
import numpy as np
import math

def calculate_sigmoid(x, b):
    sig = []
    for i in range(b):
        sigmoid = 1/(1+math.exp(-x[i][0]))
        sig.append(sigmoid)
    return sig

class NeuralNetwork:
    def three_layer(self):
        input = [[random.random() for _ in range(3)],
             [random.random() for _ in range(3)],
             [random.random() for _ in range(3)]]

        w1 = [[random.random() for _ in range(3)],
            [random.random() for _ in range(3)]]
        b1 = [random.random() for _ in range(2)]

        w2 = [[random.random() for _ in range(2)],
            [random.random() for _ in range(2)]]
        b2 = [random.random() for _ in range(2)]

        w3 = [random.random() for _ in range(2)]
        b3 = [random.random()]

        output1 = np.dot(input, np.array(w1).T) + b1
        output2 = np.dot(output1, np.array(w2).T) + b2
        output3 = np.dot(output2, np.array(w3).T) + b3

        print(output1)
        print(output2)
        print(output3)
        sig1 = 1/(1+math.exp(-output3[0]))
        sig2 = 1/(1+math.exp(-output3[1]))
        sig3 = 1/(1+math.exp(-output3[2]))

        print(f'Sigmoid: {sig1, sig2, sig3}')

class NeuralNetworkLayer:
    def initialize_weights_bias(self, neurons, inputs):
        self.weights = [[random.random() for _ in range(neurons)]
                        for _ in range(inputs)]
        self.bias = [random.random() for _ in range(neurons)]

    def feed_forward(self, input):
        self.output = np.dot(input, self.weights) + self.bias


input_layer = NeuralNetworkLayer()
hidden_layer = NeuralNetworkLayer()
output_layer = NeuralNetworkLayer()

input_layer.initialize_weights_bias(6, 8)
hidden_layer.initialize_weights_bias(3, 6)
output_layer.initialize_weights_bias(1, 3)

X = [[random.random() for _ in range(8)]
         for _ in range(3)
        ]

input_layer.feed_forward(X)
hidden_layer.feed_forward(input_layer.output)
output_layer.feed_forward(hidden_layer.output)

print(calculate_sigmoid(output_layer.output, 3))

