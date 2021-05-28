import pandas as pd 
import random
from sklearn.model_selection import train_test_split

ds = pd.read_csv('data/HIV-Activity.csv')

print(ds)

print(ds.Activity.value_counts())

neg_class = ds[ds["Activity"] == 0]
pos_class = ds[ds["Activity"] == 1]

# neg_class = random.shuffle(neg_class)
# pos_class = random.shuffle(pos_class)

neg_class_test = neg_class.sample(n=100, replace=False, random_state=42)
neg_class = neg_class.drop(neg_class_test.index)

pos_class_test = pos_class.sample(n=100, replace=False, random_state=42)
pos_class = pos_class.drop(pos_class_test.index)

print(neg_class)
print(neg_class_test)
print(pos_class)
print(pos_class_test)

train = neg_class.append(pos_class)
test = neg_class_test.append(pos_class_test)

print(train)
print(test)

train.to_csv('data/train.csv')
test.to_csv('data/test.csv')

