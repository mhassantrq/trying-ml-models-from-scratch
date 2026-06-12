"""
k means clustering from scratch

without using sklearn and numpy.
clustering of two types of data, positive and negative sentiment.
then measuring the accuracy of the algorithm.

"""

import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('data/dataset01.csv')

pos_docs = df[df['tag'] == 'positive'][:200]
neg_docs = df[df['tag'] == 'negative'][:200]

pos_words = ' '.join(pos_docs['text'].astype(str).values.flatten()).split(' ')
neg_words = ' '.join(neg_docs['text'].astype(str).values.flatten()).split(' ')

#pos_words = set([word.lower() for word in pos_words])
#neg_words = set([word.lower() for word in neg_words])

vocab = set(list(pos_words) + list(neg_words))

pos_vector = []

for doc in pos_docs['text']:
    words = doc.split(' ')
    word_count = Counter(words)
    row = [word_count[word] for word in vocab]
    pos_vector.append(row)


neg_vector = []

for doc in neg_docs['text']:
    words = doc.split(' ')
    word_count = Counter(words)
    row = [word_count[word] for word in vocab]
    neg_vector.append(row)


pos_vector = PCA(n_components=3).fit_transform(pos_vector)
neg_vector = PCA(n_components=3).fit_transform(neg_vector)

for vect in pos_vector:
    plt.scatter(vect[0], vect[1], color='red')

for vect in neg_vector:
    plt.scatter(vect[0], vect[1], color='blue')

plt.show()