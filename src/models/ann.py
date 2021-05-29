#libraries
import pandas as pd 
from sklearn import preprocessing
import numpy as np 
import pickle
from sklearn.preprocessing import OneHotEncoder 
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
import math
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras import optimizers
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input, Add, LSTM, Bidirectional, Reshape
from keras_self_attention import SeqSelfAttention
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from keras import regularizers
from keras import backend as K
import keras
from sklearn.model_selection import KFold

# GPU config for Vamsi's Laptop
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

tf.keras.backend.clear_session()
config = ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

LIMIT = 3 * 1024
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=LIMIT)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

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

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = len(np.unique(y_tot))
print(num_classes)
print("Loaded X and y")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train, y_train = shuffle(X_train, y_train, random_state=42)
print("Shuffled")

# batch size
bs = 8

# Keras NN Model
def create_model():
    input_ = Input(shape = (2622,))
    x = Dense(4, activation = 'relu')(input_)
    x = Dropout(0.2)(x)
    out = Dense(num_classes, activation = 'softmax')(x)
    classifier = Model(input_, out)

    return classifier

# training
num_epochs = 1

with tf.device('/gpu:0'):
    # model
    model = create_model()

    # adam optimizer
    opt = keras.optimizers.Adam(learning_rate = 1e-5)
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy'])

    # callbacks
    mcp_save = keras.callbacks.ModelCheckpoint('saved_models/abln.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    callbacks_list = [reduce_lr, mcp_save]

    # test and train generators
    history = model.fit(X_train, y_train, epochs = num_epochs, verbose=1, validation_data = (X_test, y_test), callbacks = callbacks_list)
    model = load_model('saved_models/abln.h5', custom_objects=SeqSelfAttention.get_custom_objects())

    print("Testing")
    y_pred_test = model.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred_test)
    print("AUC Score", auc_score)