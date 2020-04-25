import tensorflow as tf
from tensorflow.keras import layers, Sequential,Input,Model
from tensorflow.keras.layers import Embedding,LSTM
from tensorflow.keras.preprocessing.text import tokenizer_from_json
#from last_layer import CustomLayer
import os
import numpy as np
import json

"""-------------------------------------- Constants ------------------------------------------"""
GLOVE_EMBEDDING_PATH = 'D:\\Courses\\Chatbot\\glove'
OUTPUT_PATH = 'D:\\Courses\\Chatbot\\output'
MAX_SENTENCE_LENGTH = 160
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 50
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
except:
    raise FileNotFoundError("tokenizer.json file not found at output path ... !")

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
#print(M)
# Create tensors for Context and Utterance
context_input = Input(shape=(MAX_SENTENCE_LENGTH,),dtype='float32')
utterance_input = Input(shape=(MAX_SENTENCE_LENGTH,),dtype='float32')

# Encode Context and Utterance through LSTM
encoded_context = encoder(context_input)
encoded_utterance = encoder(utterance_input)
en_cont_transpose = tf.transpose(encoded_context)
generated_response = tf.matmul(encoded_context,M)

#generated_response = CustomLayer(units=256,input_dim=256)(encoded_context)

dual_encoder = Model(inputs=[context_input,utterance_input],outputs = [encoded_context,encoded_utterance,generated_response])
dual_encoder.compile(loss = 'binary_crossentropy', optimizer = 'adam')
#print(encoder.summary())
#print(dual_encoder.summary())
print(M.numpy()[0][:10])




