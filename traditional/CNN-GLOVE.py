import numpy as np
import pandas as pd
import random
import os

from gensim.models import Word2Vec
from keras import Input, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Embedding, Dense, concatenate, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Dropout, \
    Multiply
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import torch
from keras.optimizers import Adam
from keras.utils import to_categorical

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import random
from sklearn import metrics
from keras import backend as K


def read_data(fold_id, FLAG, csv_file='../data/enhancement.csv'):

    text_train, text_val, text_test = [], [], []
    y_train, y_val, y_test = [], [], []
    creator_train, creator_val, creator_test = [], [], []
    role_train, role_val, role_test = [], [], []
    freq_train, freq_val, freq_test = [], [], []

    dict_resolution = {"FIXED": 1, "INVALID": 0, "DUPLICATE": 0, "WONTFIX": 0,
                       "INCOMPLETE": 0, "WORKSFORME": 0, "EXPIRED": 0, "MOVED": 0, "INACTIVE": 0}


    df = pd.read_csv(csv_file)

    if FLAG == 0:
        test_df = df[df['fold'] == fold_id]
        val_df = df[(df['fold'].notnull()) & (df['fold'] == 2)]
        train_df = df[(df['fold'].notnull()) & (df['fold'] == 0)]
    elif FLAG == 1:
        product_list = ["Bugzilla", "SeaMonkey", "Core Graveyard", "Core", "MailNews Core",
                        "Toolkit", "Firefox", "Thunderbird", "Calendar", "Camino Graveyard"]
        test_df = df[(df['product'] == product_list[fold_id])]
        train_df = df[(df['product'] != product_list[fold_id])]
    else:
        raise ValueError("")


    for index, element in test_df.iterrows():

        text_test.append(str(element['preprocessed12345']).replace("\n", " "))
        y_test.append(dict_resolution[element['resolution']])


        if pd.isnull(element['role']):
            role_test.append(0)
        else:
            role_test.append(element['role'])
        creator_test.append(element['creator'])
        freq_test.append(int(element['creator_freq']))


    for index, element in train_df.iterrows():
        # summary_train.append(str(element['preprocessed12']).replace("\n", " "))
        text_train.append(str(element['preprocessed12345']).replace("\n", " "))
        y_train.append(dict_resolution[element['resolution']])

        if pd.isnull(element['role']):
            role_train.append(0)
        else:
            role_train.append(element['role'])
        creator_train.append(element['creator'])
        freq_train.append(int(element['creator_freq']))

    for index, element in val_df.iterrows():
        # summary_val.append(str(element['preprocessed12']).replace("\n", " "))
        text_val.append(str(element['preprocessed12345']).replace("\n", " "))
        y_val.append(dict_resolution[element['resolution']])

        if pd.isnull(element['role']):
            role_val.append(0)
        else:
            role_val.append(element['role'])
        creator_val.append(element['creator'])
        freq_val.append(int(element['creator_freq']))

    return (text_train, text_val, text_test,
            y_train, y_val, y_test,
            creator_train, creator_val, creator_test,
            role_train, role_val, role_test,
            freq_train, freq_val, freq_test)


def over_under_sample(summary_train, description_train, role_train, creator_train, freq_train, y_train,
                      over=True):
    summary, descrip,  role, creator, freq = [], [], [], [], [], []
    length = len(summary_train)
    index = np.array(list(range(length))).reshape(length, 1)
    if over:
        ros = RandomOverSampler(random_state=0)
        index, y_train = ros.fit_resample(index, y_train)
    else:
        rus = RandomUnderSampler(random_state=0)
        index, y_train = rus.fit_resample(index, y_train)
    for i in index.reshape(-1).tolist():
        summary.append(summary_train[i])
        descrip.append(description_train[i])
        role.append(role_train[i])
        creator.append(creator_train[i])
        freq.append(freq_train[i])
    summary_train = np.array(summary)
    description_train = np.array(descrip)
    role_train = np.array(role)
    creator_train = np.array(creator)
    freq_train = np.array(freq)
    return summary_train, description_train,  role_train, creator_train, freq_train, y_train


def CNN(text_train, text_val, text_test,
        y_train, y_val, y_test,
        creator_train, creator_val, creator_test,
        role_train, role_val, role_test,
        freq_train, freq_val, freq_test):
    K.clear_session()

    text_train, text_val, text_test = map(np.array, [text_train, text_val, text_test])
    y_train, y_val, y_test = map(np.array, [y_train, y_val, y_test])
    
    for dataset in [(text_train, y_train), (text_val, y_val), (text_test, y_test)]:
        idx = np.arange(len(dataset[0]))
        np.random.shuffle(idx)
        for i in range(len(dataset)):
            dataset[i][:] = dataset[i][idx]
    

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(np.concatenate((text_train, text_test), axis=0))
    word_index = tokenizer.word_index
    
    text_train_pad = pad_sequences(tokenizer.texts_to_sequences(text_train), maxlen=MAX_SEQUENCE_LENGTH_s + MAX_SEQUENCE_LENGTH_d)
    text_val_pad = pad_sequences(tokenizer.texts_to_sequences(text_val), maxlen=MAX_SEQUENCE_LENGTH_s + MAX_SEQUENCE_LENGTH_d)
    text_test_pad = pad_sequences(tokenizer.texts_to_sequences(text_test), maxlen=MAX_SEQUENCE_LENGTH_s + MAX_SEQUENCE_LENGTH_d)
    

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    embeddings_index = {}
    with open(os.path.join('../data/', 'glove.6B.100d.txt'), encoding='utf8') as f:
        for line in f:
            values = line.split()
            embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
    
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH_s + MAX_SEQUENCE_LENGTH_d, trainable=False)
    

    inputs = Input(shape=(MAX_SEQUENCE_LENGTH_s + MAX_SEQUENCE_LENGTH_d,), name='text_inputs')
    embedded_text = embedding_layer(inputs)
    conv1D = Conv1D(64, 1, padding='same', activation='relu')(embedded_text)
    conv1D = BatchNormalization()(conv1D)
    conv1D = Flatten()(conv1D)
    
    dropout = Dropout(0.5)(conv1D)
    dropout = Dense(512, activation='relu')(dropout)
    dropout = Dense(32, activation='relu')(dropout)
    predictions = Dense(1, activation='sigmoid')(dropout)
    
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, mode='auto',
                                  min_delta=0.001, cooldown=0, min_lr=0)
    
    history = model.fit(text_train_pad, y_train, batch_size=batch_size, epochs=epoch, verbose=1,
                        callbacks=[early_stopping, reduce_lr], validation_data=(text_val_pad, y_val))
    
    score = model.evaluate(text_test_pad, y_test, verbose=1)
    y_predict = model.predict(text_test_pad)
    auc = metrics.roc_auc_score(y_test, y_predict)
    y_predict = (y_predict >= 0.5).astype(int)
    precision, recall, f, _ = metrics.precision_recall_fscore_support(y_test, y_predict, beta=1.0, pos_label=1, average=None)
    cm = metrics.confusion_matrix(y_test, y_predict)
    print("Confusion Matrix:")
    print(cm)
    
    return precision, recall, f, score, auc



sg = 1
window = 7
min_count = 0
negative = 5
sample = 0.00025
hs = 1

MAX_SEQUENCE_LENGTH_s = 30
MAX_SEQUENCE_LENGTH_d = 170
EMBEDDING_DIM = 100 
epoch = 20
batch_size = 16


total_acc, total_p0, total_r0, total_f0, total_p1, total_r1, total_f1 = 0, 0, 0, 0, 0, 0, 0
FLAG = 0  

for i in range(1):
    (text_train, text_val, text_test,
            y_train, y_val, y_test,
            creator_train, creator_val, creator_test,
            role_train, role_val, role_test,
            freq_train, freq_val, freq_test) = read_data(1, FLAG, csv_file='../data/enhancement.csv')

    precision, recall, f, score, auc = CNN(text_train, text_val, text_test,
            y_train, y_val, y_test,
            creator_train, creator_val, creator_test,
            role_train, role_val, role_test,
            freq_train, freq_val, freq_test)

    total_acc += score[1]
    total_p1 += precision[1]
    total_r1 += recall[1]
    total_f1 += f[1]
    total_p0 += precision[0]
    total_r0 += recall[0]
    total_f0 += f[0]
    print("Fold {}: precision (pos) = {}, recall (pos) = {}, f1 (pos) = {}, score = {}".format(
        1, precision[1], recall[1], f[1], score))

print("Average results:")
print("Accuracy: {}".format(float(total_acc)))
print("Auc: {}".format(float(auc)))
print("Class 1 negative - Precision: {}, Recall: {}, F1: {}".format(float(total_p1), float(total_r1),
                                                           float(total_f1)))
print("Class 0 positive - Precision: {}, Recall: {}, F1: {}".format(float(total_p0), float(total_r0),
                                                           float(total_f0)))

