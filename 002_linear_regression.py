"""
linear regression from scratch

without using sklearn and numpy.
only pandas used for data reading and matplotlib for visualization.


process:
1. initialize m and b in the equation y=mx+b with 0 at start.

"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/lin_reg_multi.csv')

X = df['exp']
y = df['salary']

m=b=0

for i in range(1000000):
    pred = []
    err = []
    sum_err = 0
    sum_err_sq = 0
    
    for x in X:
        output = m*x + b
        pred.append(output)

    for i in range(len(pred)):
        diff = y[i] - pred[i]
        err.append(diff)
        sum_err += diff
        diff = diff**2
        sum_err_sq += diff

    mse = sum_err_sq / len(pred)
    sum_x_err = 0

    for i in range(len(X)):
        sum_x_err += X[i]*err[i]

    change_b = (-2/len(X)) * (sum_err)
    change_m = (-2/len(X)) * (sum_x_err)

   # print(change_b, change_m)

    lr = 0.0001

    b = b-lr*change_b
    m = m-lr*change_m

    print(m, b)

plt.scatter(X, y)

pred = []
for x in X:
    pred.append(m*x+b)

plt.plot(X, pred)
plt.show()
