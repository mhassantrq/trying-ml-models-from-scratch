"""
cosine similarity from scratch
"""
import pandas as pd

def cos_similarity(d1, d2):
    dp=0
    d1_mag=0
    d2_mag=0

    for i in range(len(d1)):
        dp += d1[i]*d2[i]
        d1_mag += d1[i]**2
        d2_mag += d2[i]**2

    d1_mag = d1_mag ** 0.5
    d2_mag = d2_mag ** 0.5

    cos_sim = dp / (d1_mag*d2_mag)
    return round(cos_sim, 2)


df = pd.read_csv('data/dataset01.csv')
df = df['text'][0:10]
docs = []
vocab=[]

for d in df:
    d=d.lower().split()
    docs.append(d)

for doc in docs:
    for w in doc:
        if w not in vocab:
            vocab.append(w)

doc_vect = [[] for _ in range(len(docs))]

for w in vocab:
    for i in range(len(docs)):
        if w in docs[i]:
            doc_vect[i].append(1)
        else:
            doc_vect[i].append(0)

sim_vect = [[] for _ in range(len(doc_vect))]

for i in range(len(doc_vect)):
    for j in range(len(doc_vect)):
        sim_vect[i].append(cos_similarity(doc_vect[i], doc_vect[j]))

for i in range(len(sim_vect)):
    print(f'doc {i+1}: {sim_vect[i]}')
