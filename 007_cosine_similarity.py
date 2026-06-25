"""
cosine similarity from scratch
"""

d1 = [1,2,3,5]
d2 = [2,5,0,6]

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

print(f'similarity index: {cos_sim}')