"""
knn from scratch
"""

import math

k=3

def euclidean_dist(x,y):
    sum=0
    for i in range(len(x)):
        sum += (x[i] - y[i])**2
    return math.sqrt(sum)

