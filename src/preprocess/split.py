import pandas as pd 
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

ds = pd.read_csv('../../data/HIV-Activity.csv')
ds = ds.drop_duplicates()
ds = shuffle(ds)
# print(ds)

# print(ds.Activity.value_counts())

neg_class = ds[ds["Activity"] == 0]
pos_class = ds[ds["Activity"] == 1]
lis_ng_class = len(list(neg_class["Activity"]))
print(lis_ng_class)
# neg_class = random.shuffle(neg_class)
# pos_class = random.shuffle(pos_class)

neg_class_test = neg_class[0:300]
neg_class = neg_class[300:]

pos_class_test = pos_class[0:300]
pos_class = pos_class[300:lis_ng_class]

# print(neg_class)
# print(neg_class_test)
# print(pos_class)
# print(pos_class_test)

train = neg_class.append(pos_class)
test = neg_class_test.append(pos_class_test)

print(train)
print(test)

train.to_csv('../../data/train.csv')
test.to_csv('../../data/test.csv')

'''
600 test
2286 train
'''