import tensorflow as tf
from tensorflow.keras import layers, Sequential,Input,Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Embedding,LSTM,Dense
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import os
import numpy as np
import json
from preprocessing import read_train_TFRecords

"""-------------------------------------- Constants ------------------------------------------"""
GLOVE_EMBEDDING_PATH = 'D:\\Courses\\Chatbot\\glove'
OUTPUT_PATH = 'D:\\Courses\\Chatbot\\output'
MAX_SENTENCE_LENGTH = 160
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 50
BATCH_SIZE = 1
SHUFFLE_BUFFER = 256
"""------------------------------------------------------------------------------------------"""

print("Starting to build model ...")

print("Indexing word vectors ...")

embeddings_index = dict()

# Map words to their embeddings
with open(os.path.join(GLOVE_EMBEDDING_PATH,"glove.6B.50d.txt"),'r',encoding='utf-8') as glove_file:
    for row in glove_file:
        values = row.split()
        word = values[0]
        try:
            embedding = np.asarray(values[1:],dtype = 'float32')
        except ValueError:
            continue
        embeddings_index[word] = embedding

print("Maximum Length of Sentence :{}".format(MAX_SENTENCE_LENGTH))
print("Maximum Number of Words :{}".format(MAX_NB_WORDS))

try:
    with open(os.path.join(OUTPUT_PATH,"tokenizer.json")) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
except Exception:
    print("tokenizer.json file not found at output path ... !")

word_index = tokenizer.word_index
num_words = min(MAX_NB_WORDS,len(word_index)) + 1       # Why + 1 ??
EMBEDDING_MATRIX = np.zeros((num_words,EMBEDDING_DIM))

for word,index in word_index.items():
    if index > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)       #.get() used because for key not found it returns None and not KeyError !
    if embedding_vector is not None:
        EMBEDDING_MATRIX[index] = embedding_vector

print("Building Dual Encoder LSTM ...")

encoder = Sequential()
encoder.add(Embedding(input_dim = MAX_NB_WORDS,output_dim = EMBEDDING_DIM,input_length = MAX_SENTENCE_LENGTH))
encoder.add(LSTM(units = 256))
M_init = tf.random_normal_initializer()
M = tf.Variable(initial_value = M_init(shape = (256,256)),trainable = True)

# Create tensors for Context and Utterance
context_input = Input(shape=(MAX_SENTENCE_LENGTH,),dtype='float32')
utterance_input = Input(shape=(MAX_SENTENCE_LENGTH,),dtype='float32')

# Encode Context and Utterance through LSTM
encoded_context = encoder(context_input)            # Shape = (None,256)
encoded_utterance = encoder(utterance_input)        # Actual response encoding (None,256) --> Need to take its transpose to make dimenions add up

generated_response = tf.matmul(encoded_context,M)   # Shape = (None,256)
projection = tf.matmul(generated_response,tf.transpose(encoded_utterance))
probability = tf.math.sigmoid(projection)

dual_encoder = Model(inputs=[context_input,utterance_input],outputs = probability)
plot_model(dual_encoder, os.path.join(OUTPUT_PATH,'my_first_model.png'),show_shapes = True)

dual_encoder.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop',metrics=['accuracy'])

print(M.numpy()[0][:2])


def create_batched_dataset(data_path):
    tfrecord_dataset = tf.data.TFRecordDataset(os.path.join(data_path,"train.tfrecords"))
    parsed_dataset = tfrecord_dataset.map(read_train_TFRecords,num_parallel_calls = 8)
    parsed_dataset = parsed_dataset.repeat()
    parsed_dataset = parsed_dataset.shuffle(SHUFFLE_BUFFER)
    parsed_dataset = parsed_dataset.batch(BATCH_SIZE)
    iterator = tf.compat.v1.data.make_one_shot_iterator(parsed_dataset)
    batched_context,batched_utterance,batched_labels = iterator.get_next()
    return batched_context,batched_utterance,batched_labels

batched_context,batched_utterance,batched_labels = create_batched_dataset(OUTPUT_PATH)
print(tf.compat.v1.trainable_variables)
for i in range(50):
    dual_encoder.fit([batched_context,batched_utterance],batched_labels,batch_size = BATCH_SIZE,epochs = 1)
    print(M.numpy()[0][:2])
    #print("Epoch {} completed ...!".format(i))




