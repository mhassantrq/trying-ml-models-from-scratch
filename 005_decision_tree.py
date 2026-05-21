"""
decision tree from scratch. example assumes only two final decisions
"""

import pandas as pd
from collections import Counter


def gini_index(X, y):
    bound = (X.max() + X.min())/2
    result = Counter(y)
    gi_before = 1 - ((result['F'] / sum(result.values()))**2 + (result['P'] / sum(result.values()))**2)

    for i in range(len(X)):
        print(X.iloc[i])


df = pd.read_csv('data/classification01.csv')
gini_index(df['assignments'], df['result'])
