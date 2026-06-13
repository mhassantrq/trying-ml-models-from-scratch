"""
pending...
Linear Regression from scratch
"""

import pandas as pd

df = pd.read_csv('data/lin_reg_multi.csv')

X = df['exp']
y = df['salary']

m=b=0

for i in range(5):
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

    lr = 0.01

    b = b-lr*change_b
    m = m-lr*change_m

    print(m, b, mse)

