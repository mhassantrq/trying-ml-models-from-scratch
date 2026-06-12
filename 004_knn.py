"""
basic variant. advanced pending...
knn from scratch
"""

import math
import matplotlib.pyplot as plt

def euclidean_dist(x,y):
    sum=0
    for i in range(len(x)):
        sum += (x[i] - y[i])**2
    return math.sqrt(sum)

points = {
    'group1': [[3,5], [4,5], [3,6], [2,6]],
    'group2': [[7,3], [5,5], [7,5], [7,6], [6,3]],
}

x1,y1 = zip(*points['group1'])
x2,y2 = zip(*points['group2'])

plt.scatter(x1,y1)
plt.scatter(x2,y2)

new_point = [3,3]

plt.scatter(3,3)
plt.show()

nn_dist = []
for group in points:
    for point in points[group]:
        dist = euclidean_dist(point, new_point)
        nn_dist.append([dist, group])

print(sorted(nn_dist))