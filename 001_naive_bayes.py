"""
Naive Bayes from scratch
"""

import pandas as pd
from collections import defaultdict

df = pd.read_csv('data/dataset01.csv')
df['tag'] = df['tag'].map({'positive': 1, 'negative': 0})

vocab = defaultdict(int)
pos_bagofwords = defaultdict(int)
neg_bagofwords = defaultdict(int)

for_vocab = ' '.join(df['text'].astype(str).values.flatten()).split(' ')

for word in for_vocab:
    vocab[word] += 1

print(f'Total Words at start: {len(for_vocab)}')
print(f'Words in vocabulary without preprocessing: {len(vocab)}')

neg_count = (df['tag'] == 0).sum()
pos_count = (df['tag'] == 1).sum()

print(f'Negative Sentiments: {neg_count}, Positive Sentiments: {pos_count}')

neg_prior = neg_count / (neg_count+pos_count)
pos_prior = pos_count / (neg_count+pos_count)

print(f'neg prior probability: {neg_prior}, pos prior probability: {pos_prior}')

neg_rows = df[df['tag'] == 0]
neg_words = ' '.join(neg_rows['text'].astype(str).values.flatten()).split(' ')

pos_rows = df[df['tag'] == 1]
pos_words = ' '.join(pos_rows['text'].astype(str).values.flatten()).split(' ')

print(f'before preprocessing, pos words: {len(pos_words)}, neg words: {len(neg_words)}')

for w in neg_words:
    neg_bagofwords[w.lower()] += 1

for w in pos_words:
    pos_bagofwords[w.lower()] += 1

print(f'after normalization, pos words: {len(pos_bagofwords)}, neg words: {len(neg_bagofwords)}')