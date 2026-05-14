"""
Naive Bayes from scratch
"""

import pandas as pd
from collections import defaultdict

vocab = defaultdict(int)
neg_bagofwords = defaultdict(int)
pos_bagofwords = defaultdict(int)

pos_cond_prob = defaultdict(int)
neg_cond_prob = defaultdict(int)

df = pd.read_csv('data/dataset01.csv')

total_neg_rows = df[df['tag'] == 'negative']
total_pos_rows = df[df['tag'] == 'positive']

print(f'before train test split. neg rows: {len(total_neg_rows)}, pos rows: {len(total_pos_rows)}')

pos_rows_random = total_pos_rows.sample(frac=1)
neg_rows_random = total_neg_rows.sample(frac=1)

pos_train_count = int(len(total_pos_rows)*0.8)
neg_train_count = int(len(total_neg_rows)*0.8)

train_pos_rows = pos_rows_random.iloc[:pos_train_count]
test_pos_rows = pos_rows_random.iloc[pos_train_count:]

train_neg_rows = neg_rows_random.iloc[:neg_train_count]
test_neg_rows = neg_rows_random.iloc[neg_train_count:]

print(f'neg rows train: {len(train_neg_rows)}, test: {len(test_neg_rows)}')
print(f'pos rows train: {len(train_pos_rows)}, test: {len(test_pos_rows)}')


for_vocab = ' '.join(train_pos_rows.astype(str).values.flatten()).split(' ')
for_vocab = for_vocab + ' '.join(train_neg_rows.astype(str).values.flatten()).split(' ')

print(len(for_vocab))

for word in for_vocab:
    vocab[word.lower()] += 1

print(f'Total Words at start: {len(for_vocab)}')
print(f'Words in vocabulary without preprocessing: {len(vocab)}')

neg_prior = len(train_neg_rows) / (len(train_neg_rows) + len(train_pos_rows))
pos_prior = len(train_pos_rows) / (len(train_neg_rows) + len(train_pos_rows))

print(f'neg prior probability: {neg_prior}, pos prior probability: {pos_prior}')

neg_words = ' '.join(train_neg_rows['text'].astype(str).values.flatten()).split(' ')
pos_words = ' '.join(train_pos_rows['text'].astype(str).values.flatten()).split(' ')

print(f'before preprocessing, pos words: {len(pos_words)}, neg words: {len(neg_words)}')

for w in neg_words:
    neg_bagofwords[w.lower()] += 1

for w in pos_words:
    pos_bagofwords[w.lower()] += 1

print(f'after normalization, pos words: {len(pos_bagofwords)}, neg words: {len(neg_bagofwords)}')

for w in vocab:
    neg_cond_prob[w] = float((neg_bagofwords[w] + 1) / (len(vocab) + len(neg_bagofwords)))
    pos_cond_prob[w] = float((pos_bagofwords[w] + 1) / (len(vocab) + len(pos_bagofwords)))
