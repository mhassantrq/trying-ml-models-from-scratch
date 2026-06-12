"""
k means clustering from scratch

without using sklearn.
clustering of two types of data, positive and negative sentiment.
then measuring the accuracy of the algorithm.

the testing below is done on smaller dataset to avoid long processing delay.
"""

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import math

df = pd.read_csv('data/dataset01.csv')

pos_docs = df[df['tag'] == 'positive'][:100]
neg_docs = df[df['tag'] == 'negative'][:100]

pos_words = ' '.join(pos_docs['text'].astype(str).values.flatten()).split(' ')
neg_words = ' '.join(neg_docs['text'].astype(str).values.flatten()).split(' ')

vocab = set(list(pos_words) + list(neg_words))

pos_vector = []
test_pos_vector = []
i=0

for doc in pos_docs['text']:
    i+=1
    words = doc.split(' ')
    word_count = Counter(words)
    row = [word_count[word] for word in vocab]
    if i<= 80:
        pos_vector.append(row)
    else:
        test_pos_vector.append(row)

neg_vector = []
test_neg_vector = []
i=0

for doc in neg_docs['text']:
    i+=1
    words = doc.split(' ')
    word_count = Counter(words)
    row = [word_count[word] for word in vocab]
    if i<= 80:
        neg_vector.append(row)
    else:
        test_neg_vector.append(row)

pos_centroid = []
neg_centroid = []
for i in range(len(vocab)):
    pos_temp_sum = 0
    neg_temp_sum = 0
    for vect in pos_vector:
        pos_temp_sum += vect[i]
    pos_temp_sum = pos_temp_sum/len(pos_vector)
    pos_centroid.append(pos_temp_sum)
    for vect in neg_vector:
        neg_temp_sum += vect[i]
    neg_temp_sum = neg_temp_sum/len(neg_vector)
    neg_centroid.append(neg_temp_sum)


tp=fp=tn=fn=0

for doc in test_pos_vector:
    pos_euclidean = 0
    neg_euclidean = 0
    for i in range(len(doc)):
        pos_euclidean += (doc[i] - pos_centroid[i]) ** 2
        neg_euclidean += (doc[i] - neg_centroid[i]) ** 2

    pos_euclidean = math.sqrt(pos_euclidean)
    neg_euclidean = math.sqrt(neg_euclidean)

    if pos_euclidean <= neg_euclidean:
        tp+=1
    else:
        fn+=1


for doc in test_neg_vector:
    pos_euclidean = 0
    neg_euclidean = 0
    for i in range(len(doc)):
        pos_euclidean += (doc[i] - pos_centroid[i]) ** 2
        neg_euclidean += (doc[i] - neg_centroid[i]) ** 2

    pos_euclidean = math.sqrt(pos_euclidean)
    neg_euclidean = math.sqrt(neg_euclidean)

    if neg_euclidean <= pos_euclidean:
        tn+=1
    else:
        fp+=1

print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')

acc = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2 * ((precision*recall)/(precision+recall))

print(f'Accruacy: {acc:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1_score:.2f}')


# pos_vector = PCA(n_components=3).fit_transform(pos_vector)
# neg_vector = PCA(n_components=3).fit_transform(neg_vector)

# for vect in pos_vector:
#     plt.scatter(vect[0], vect[1], color='red')

# for vect in neg_vector:
#     plt.scatter(vect[0], vect[1], color='blue')

# plt.show()