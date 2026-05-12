"""
Naive Bayes from scratch
"""

import pandas as pd


df = pd.read_csv('data/dataset01.csv')
df['tag'] = df['tag'].map({'positive': 1, 'negative': 0})




print(df)