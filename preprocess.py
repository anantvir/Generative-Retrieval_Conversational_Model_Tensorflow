import tensorflow as tf
import pickle
import os
import io
import json
import csv
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

"""D:\\Courses\\Chatbot\\dataset\\data\\, D:\\Courses\\Chatbot\\dataset_trimmed\\data\\"""
DIRNAME_ABSOLUTE = os.path.dirname(__file__)
DATA_DIR = os.path.join(DIRNAME_ABSOLUTE, 'dataset\\data\\')
OUTPUT_PATH = os.path.join(DIRNAME_ABSOLUTE, 'output')
VOCAB_SIZE = 20000
EMBEDDING_DIMENSION = 128
MAX_LENGTH_OF_SENTENCE = 160
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'
OOV_TOKEN = '<OOV>'

def build_train_set():
    print("Building train data ...")
    train = pd.read_csv(DATA_DIR + 'train.csv')
    rows = train.shape[0]
    context = []
    response = []
    label = []
    for i in range(rows):
        context.append(train.loc[i].Context)
        response.append(train.loc[i].Utterance)
        label.append(int(train.loc[i].Label))
    return context, response, label

def build_prediction_set(csv_file):
    df = pd.read_csv(DATA_DIR + csv_file)
    rows = df.shape[0]
    context = []
    response = []
    label = []
    for i in range(rows):
        # add 10 contexts
        for j in range(10):
            context.append(df.loc[i].Context)
        # add ground truth
        response.append(df.loc[i]['Ground Truth Utterance'])
        # add 9 false responses
        response.append(df.loc[i].Distractor_0)
        response.append(df.loc[i].Distractor_1)
        response.append(df.loc[i].Distractor_2)
        response.append(df.loc[i].Distractor_3)
        response.append(df.loc[i].Distractor_4)
        response.append(df.loc[i].Distractor_5)
        response.append(df.loc[i].Distractor_6)
        response.append(df.loc[i].Distractor_7)
        response.append(df.loc[i].Distractor_8)
        # add labels now 1, followed by nine 0
        label.append(1) # ground
        label.append(0) # 0
        label.append(0) # 1
        label.append(0) # 2
        label.append(0) # 3
        label.append(0) # 4
        label.append(0) # 5
        label.append(0) # 6
        label.append(0) # 7 
        label.append(0) # 8
    return context, response, label

def main():

    # loading train data
    train_c, train_r, train_l = build_train_set() 

    # loading test data
    print("building prediction set ...")
    test_c, test_r, test_l = build_prediction_set('test.csv')
    
    # loading dev data
    print("Building dev set ...")
    dev_c, dev_r, dev_l = build_prediction_set('valid.csv')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_c)
    tokenizer.fit_on_texts(train_r)
    tokenizer.fit_on_texts(test_c)
    tokenizer.fit_on_texts(test_r)
    tokenizer.fit_on_texts(dev_c)
    tokenizer.fit_on_texts(dev_r)

    train_c = tokenizer.texts_to_sequences(train_c)
    train_r = tokenizer.texts_to_sequences(train_r)
    test_c = tokenizer.texts_to_sequences(test_c)
    test_r = tokenizer.texts_to_sequences(test_r)
    dev_c = tokenizer.texts_to_sequences(dev_c)
    dev_r = tokenizer.texts_to_sequences(dev_r)

    #MAX_SEQUENCE_LENGTH = max([len(seq) for seq in train_c + train_r
                                                    #+ test_c + test_r
                                                    #+ dev_c + dev_r])
    
    MAX_NB_WORDS = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index
    MAX_SEQUENCE_LENGTH = 160
    print("MAX_SEQUENCE_LENGTH: {}".format(MAX_SEQUENCE_LENGTH))
    print("MAX_NB_WORDS: {}".format(MAX_NB_WORDS))

    train_c = pad_sequences(train_c, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
    train_r = pad_sequences(train_r, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
    test_c = pad_sequences(test_c, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
    test_r = pad_sequences(test_r, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
    dev_c = pad_sequences(dev_c, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
    dev_r = pad_sequences(dev_r, maxlen=MAX_SEQUENCE_LENGTH,padding='post')

    # shuffle training set
    indices = np.arange(train_c.shape[0])
    
    np.random.shuffle(indices)

    train_c = np.asarray(train_c)
    train_r = np.asarray(train_r)
    train_l = np.asarray(train_l)

    train_c = train_c[indices]
    train_r = train_r[indices]
    train_l = train_l[indices]

    pickle.dump([train_c, train_r, train_l], open('D:\\Courses\\Chatbot\\blog\\output\\' + "train.pkl", "wb"), protocol=-1)
    pickle.dump([test_c, test_r, test_l], open('D:\\Courses\\Chatbot\\blog\\output\\' + "test.pkl", "wb"), protocol=-1)
    pickle.dump([dev_c, dev_r, dev_l], open('D:\\Courses\\Chatbot\\blog\\output\\' + "dev.pkl", "wb"), protocol=-1)

    pickle.dump([MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, word_index], open('D:\\Courses\\Chatbot\\blog\\output\\' + "params.pkl", "wb"), protocol=-1)
    

main()