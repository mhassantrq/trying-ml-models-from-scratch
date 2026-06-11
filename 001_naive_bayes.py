"""
Naive Bayes from scratch.

The entire logic for naive bayes has been written from scratch.
The vectorization of features, stopword removal, calculation of bag of words, probabilitites both prior and conditional.
Moreover, accuracy, precision, recall and f1 score have been calculated without sklearn or any other module.


The only libraries or modules used in the code are listed below with their intended use:
1. pandas for loading dataset
2. defaultdict for storing data
3. math for using math.log while multiplying probabilities
"""

import pandas as pd
from collections import defaultdict
import math

vocab = defaultdict(int)
neg_bagofwords = defaultdict(int)
pos_bagofwords = defaultdict(int)

pos_cond_prob = defaultdict(int)
neg_cond_prob = defaultdict(int)

tp=tn=fp=fn=0

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

stopword_df = pd.read_csv('data/stopwords.csv')
stopwords = ' '.join(stopword_df['words'].astype(str).values.flatten()).split(' ')

print(len(for_vocab))

for word in for_vocab:
    if word.lower() not in stopwords:
        vocab[word.lower()] += 1

print(f'Total Words at start: {len(for_vocab)}')
print(f'Words in vocabulary without preprocessing: {len(vocab)}')

neg_prior = len(train_neg_rows) / (len(train_neg_rows) + len(train_pos_rows))
pos_prior = len(train_pos_rows) / (len(train_neg_rows) + len(train_pos_rows))

neg_prior = math.log(neg_prior)
pos_prior = math.log(pos_prior)

print(f'neg prior probability: {neg_prior}, pos prior probability: {pos_prior}')

neg_words = ' '.join(train_neg_rows['text'].astype(str).values.flatten()).split(' ')
pos_words = ' '.join(train_pos_rows['text'].astype(str).values.flatten()).split(' ')

print(f'before preprocessing, pos words: {len(pos_words)}, neg words: {len(neg_words)}')

stopword_df = pd.read_csv('data/stopwords.csv')
stopwords = ' '.join(stopword_df['words'].astype(str).values.flatten()).split(' ')

for w in neg_words:
    if w.lower() not in stopwords:
        neg_bagofwords[w.lower()] += 1

for w in pos_words:
    if w.lower() not in stopwords:
        pos_bagofwords[w.lower()] += 1

print(f'after normalization, pos words: {len(pos_bagofwords)}, neg words: {len(neg_bagofwords)}')

for w in vocab:
    neg_cond_prob[w.lower()] = float((neg_bagofwords[w.lower()] + 1) / (len(vocab) + len(neg_bagofwords)))
    pos_cond_prob[w.lower()] = float((pos_bagofwords[w.lower()] + 1) / (len(vocab) + len(pos_bagofwords)))

"""
Testing
"""

for row in test_neg_rows['text']:
    words = row.split(' ')
    prob_neg = neg_prior
    prob_pos = pos_prior
    for word in words:
        if word.lower() in neg_cond_prob:
            prob_neg += math.log(neg_cond_prob[word.lower()])
        if word.lower() in pos_cond_prob:
            prob_pos += math.log(pos_cond_prob[word.lower()])
    if prob_neg >= prob_pos:
        tn += 1
    else:
        fp += 1


for row in test_pos_rows['text']:
    words = row.split(' ')
    prob_neg = neg_prior
    prob_pos = pos_prior
    for word in words:
        if word.lower() in neg_cond_prob:
            prob_neg += math.log(neg_cond_prob[word.lower()])
        if word.lower() in pos_cond_prob:
            prob_pos += math.log(pos_cond_prob[word.lower()])
    if prob_pos >= prob_neg:
        tp += 1
    else:
        fn += 1

print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')

acc = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2 * ((precision*recall)/(precision+recall))

print(f'Accruacy: {acc:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1_score:.2f}')


"""
preprocessing pending.
top_words_vocab = sorted(vocab.items(), key=lambda word: word[1], reverse=True)[:10]
print(top_words_vocab)

as can be seen that many words with most counts are stop words, stemming is required and much more. will come back to this algorithm soon.
"""

