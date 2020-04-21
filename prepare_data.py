import tensorflow as tf
import pickle
import os
import io
import json
import csv
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""--------------------------------- Constants -------------------------------"""
"""D:\\Courses\\Chatbot\\dataset\\data\\, D:\\Courses\\Chatbot\\dataset_trimmed\\data\\"""
DATA_DIRECTORY_PATH = 'D:\\Courses\\Chatbot\\dataset_trimmed\\data\\'
OUTPUT_PATH = 'D:\\Courses\\Chatbot\\output'
VOCAB_SIZE = 10000
EMBEDDING_DIMENSION = 128
MAX_LENGTH_OF_SENTENCE = 160
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'
OOV_TOKEN = '<OOV>'

"""---------------------------------------------------------------------------"""

def read_train_data(data_path):
    context_train = []
    utterance_train = []
    labels = []
    with open(os.path.join(data_path,'train.csv'),encoding='utf8') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            context_train.append(row[0])
            utterance_train.append(row[1])
            labels.append(int(row[2]))
    return context_train,utterance_train,labels

def read_test_data(data_path):
    context_test = ground_truth = distractor_0 = distractor_1 = distractor_2 = distractor_3 = distractor_4 = distractor_5 = distractor_6 = distractor_7 = distractor_8 = []
    with open(os.path.join(data_path,'test.csv')) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            context_test.append(row[0])
            ground_truth.append(row[1])
            distractor_0.append(row[2])
            distractor_1.append(row[3])
            distractor_2.append(row[4])
            distractor_3.append(row[5])
            distractor_4.append(row[6])
            distractor_5.append(row[7])
            distractor_6.append(row[8])
            distractor_7.append(row[9])
            distractor_8.append(row[10])
    return context_test,ground_truth,distractor_0,distractor_1,distractor_2,distractor_3,distractor_4,distractor_5,distractor_6,distractor_7,distractor_8

def extract_and_store_vocabulary(vocab_size,oov_token,context_column,utterance_column):
    tokenizer = Tokenizer(num_words = vocab_size,oov_token = oov_token)
    tokenizer.fit_on_texts(context_column + utterance_column)

    # Dictionary in the format {word : index}
    word_index = tokenizer.word_index
    
    # Saving Vocab to JSON in the format {word : mapped integer}
    tokernizer_json_string = tokenizer.to_json()
    with io.open(os.path.join(OUTPUT_PATH,'tokenizer.json'),'w',encoding='utf-8') as f:
        f.write(json.dumps(tokernizer_json_string,ensure_ascii=False))

    return tokenizer

def map_text_to_integers_and_pad_train(tokenizer,context,utterance):

    integer_context_sequences = tokenizer.texts_to_sequences(context)
    integer_utterance_sequences = tokenizer.texts_to_sequences(utterance)

    context_sequences_padded = pad_sequences(integer_context_sequences,maxlen = MAX_LENGTH_OF_SENTENCE,padding = PADDING_TYPE, truncating = TRUNC_TYPE)
    utterance_sequences_padded = pad_sequences(integer_utterance_sequences,maxlen = MAX_LENGTH_OF_SENTENCE,padding = PADDING_TYPE, truncating = TRUNC_TYPE)

    return (context_sequences_padded,utterance_sequences_padded)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_train_Example(context,utterance,label,tokenizer):
    context_feature = _bytes_feature(context)
    utterance_feature = _bytes_feature(utterance)
    label_feature = _int64_feature(label)
    feature = {
        "Context" : context_feature,
        "Utterance" : utterance_feature,
        "Label" : label_feature
    }
    example_proto = tf.train.Example(features = tf.train.Features(feature = feature))
    return example_proto.SerializeToString()

def write_train_tfRecords(output_path,serialized_example):
    filename = 'train.tfrecords'
    with tf.io.TFRecordWriter(os.path.join(output_path,filename)) as writer:
        writer.write(serialized_example)


context_col,utterrance_col,label_col = read_train_data(DATA_DIRECTORY_PATH)
tokenizer = extract_and_store_vocabulary(VOCAB_SIZE,OOV_TOKEN,context_col,utterrance_col)
context_padded,utterance_padded = map_text_to_integers_and_pad_train(tokenizer,context_col,utterrance_col)
labels = np.array(label_col)

dataset = tf.data.Dataset.from_tensor_slices((context_padded,utterance_padded,labels))

def create_train_TFRecords(dataset,output_path):
    file_name = 'train.tfrecords'
    print("Creating train TFRecords ...")
    with tf.io.TFRecordWriter(os.path.join(output_path,file_name)) as writer:
        for context_tensor,utterance_tensor,label_tensor in dataset:
            serialized_example = create_train_Example(tf.io.serialize_tensor(context_tensor),tf.io.serialize_tensor(utterance_tensor),label_tensor.numpy(),tokenizer)
            writer.write(serialized_example)
    print("Train TFRecords created successfully !")

print("Hello")