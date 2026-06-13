"""
pending...
Linear Regression from scratch
"""

import pandas as pd

df = pd.read_csv('data/lin_reg_multi.csv')

X = df['exp']
y = df['salary']

pred = []
m=b=0
output = 0

for x in X:
    output = m*x + b
    pred.append(output)

mse = 0
err = []

for i in range(len(pred)):
    err = X[i] - pred[i]
    mse += err

mse = mse / len(pred)

print(mse)
