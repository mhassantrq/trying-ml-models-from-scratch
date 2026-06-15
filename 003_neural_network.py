"""
Neural Network from scratch

This code is still pending changes, unless this comment line is removed.
"""

import random
import numpy as np
import math

#random.seed(0)
#np.random.seed(0)

def calculate_sigmoid(x, b):
    sig = []
    for i in range(b):
        sigmoid = 1/(1+math.exp(-x[i][0]))
        sig.append(sigmoid)
    return sig

def calculate_relu(x, b):
    relu = []
    for i in range(b):
        relu.append(max(0,x[i]))
    return relu


# this class is for basic explanation only.
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

class InputLayer:
    def feed_forward(self, input):
        self.output = input

class NeuralNetworkLayer:
    def initialize_weights_bias(self, inputs, neurons):
        self.weights = np.random.uniform(-1,1,(inputs, neurons))
        self.bias = np.random.uniform(-1,1,neurons)

    def feed_forward(self, input):
        self.output = np.dot(input, self.weights) + self.bias

input_layer = InputLayer()
hidden_layer1 = NeuralNetworkLayer()
hidden_layer2 = NeuralNetworkLayer()
output_layer = NeuralNetworkLayer()

X = np.random.uniform(-1,1,(3, 3))

hidden_layer1.initialize_weights_bias(3, 3)
hidden_layer2.initialize_weights_bias(3, 3)
output_layer.initialize_weights_bias(3, 1)

input_layer.feed_forward(X)
hidden_layer1.feed_forward(input_layer.output)
hidden_layer2.feed_forward(hidden_layer1.output)
output_layer.feed_forward(hidden_layer2.output)

print(calculate_sigmoid(output_layer.output, 3))

