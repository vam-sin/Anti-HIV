#libraries
import pandas as pd 
from sklearn import preprocessing
import numpy as np 
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
import math
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

# dataset import
# train 
ds_train = pd.read_csv('../../data/train.csv')

y_train = list(ds_train["Activity"])

X_train = list(ds_train["SMILES"])

# test
ds_test = pd.read_csv('../../data/test.csv')

y_test = list(ds_test["Activity"])

X_test = list(ds_test["SMILES"])

tokenizer = Tokenizer()

X_tot = []

for i in range(len(X_train)):
    X_tot.append(X_train[i])

for i in range(len(X_test)):
    X_tot.append(X_test[i])

tokenizer.fit_on_texts(X_tot)

X_train = tokenizer.texts_to_matrix(X_train)
X_test = tokenizer.texts_to_matrix(X_test)

# y process
y_tot = []

for i in range(len(y_train)):
    y_tot.append(y_train[i])

for i in range(len(y_test)):
    y_tot.append(y_test[i])

le = preprocessing.LabelEncoder()
le.fit(y_tot)

y_train = np.asarray(le.transform(y_train))
y_test = np.asarray(le.transform(y_test))

num_classes = len(np.unique(y_tot))
print(num_classes)
print("Loaded X and y")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


X_train, y_train = shuffle(X_train, y_train, random_state=42)
print("Shuffled")

# logreg model
clf = LogisticRegression(solver='liblinear', random_state=0).fit(X_train, y_train)
test_score = clf.score(X_test, y_test)

y_pred = clf.predict(X_test)

print("AUC Score: ", roc_auc_score(y_test, y_pred))
print("Test Score: ", test_score)

