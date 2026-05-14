"""
Basic Neural Network from scratch
"""

#   below code is for testing only. will be removed later

import random


x1 = random.random()
w1 = random.random()
x2 = random.random()
w2 = random.random()
b = random.random()

res = x1*w1 + x2*w2 + b

res = 1 / (1 + 2.718 ** -res)
print(res)