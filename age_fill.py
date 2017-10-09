import pandas as pd
import numpy as np
dataset = pd.read_csv('dataset/train.csv')
print(dataset.head(n=5))
sum1,count=0,0
for a in dataset['Age']:
    if  np.isnan(a):
        continue
    else:
        count = count +1
        sum1 = sum1 +a
        #print sum1
mean = sum1 / count
print mean
# for a in dataset['Age']:
#     if np.isnan(a):
#         dataset[[a]]= mean
def prepare_age(row):
    if(np.isnan(row["Age"])):
        row["Age"] = mean
    return row

dataset = dataset.apply(lambda row: prepare_age(row), axis=1)
