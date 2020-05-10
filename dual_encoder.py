"""
Motivations for this project:
1) https://basmaboussaha.wordpress.com/2017/10/18/implementation-of-dual-encoder-using-keras/comment-page-1/?unapproved=229&moderation-hash=f6485213fcee44e23ad52e4c5c231424#comment-229
2) A great blog post by Denny Britz using Estimator API --> http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/
"""

import tensorflow as tf
from tensorflow.keras import layers, Sequential,Input,Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Embedding,LSTM,Dense
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import RMSprop
import os
import numpy as np
import json
from callbacks import Histories
import pickle
from helpers import compute_recall_ks,str2bool,recall

"""-------------------------------------- Constants ------------------------------------------"""
DIRNAME_ABSOLUTE = os.path.dirname(__file__)
GLOVE_EMBEDDING_PATH = os.path.join(DIRNAME_ABSOLUTE, 'glove')
OUTPUT_PATH = os.path.join(DIRNAME_ABSOLUTE, 'output')
EMBEDDING_DIM = 100
BATCH_SIZE = 256
"""------------------------------------------------------------------------------------------"""

print("Building Model ...")

print("Mapping words to Embeddings ...")

embeddings_index = dict()

# Map words to their embeddings
with open(os.path.join(GLOVE_EMBEDDING_PATH,"glove.6B.100d.txt"),'r',encoding='utf-8') as glove_file:
	for row in glove_file:
		values = row.split()
		word = values[0]
		try:
			embedding = np.asarray(values[1:],dtype = 'float32')
		except ValueError:
			continue
		embeddings_index[word] = embedding

MAX_SENTENCE_LENGTH, MAX_NB_WORDS, word_index = pickle.load(open(os.path.join(OUTPUT_PATH,'params.pkl'), 'rb'))
print(MAX_NB_WORDS)
print("Maximum Length of Sentence :{}".format(MAX_SENTENCE_LENGTH))
print("Maximum Number of Words :{}".format(MAX_NB_WORDS))

num_words = min(MAX_NB_WORDS,len(word_index)) + 1       # +1 for <UNK>
EMBEDDING_MATRIX = np.zeros((num_words,EMBEDDING_DIM))

for word,index in word_index.items():
	if index > MAX_NB_WORDS:
		continue
	embedding_vector = embeddings_index.get(word)       #.get() used because for key not found it returns None and not KeyError !
	if embedding_vector is not None:
		EMBEDDING_MATRIX[index] = embedding_vector

print("Building Dual Encoder LSTM ...")

encoder = Sequential()
encoder.add(Embedding(input_dim = MAX_NB_WORDS,output_dim = EMBEDDING_DIM,input_length = MAX_SENTENCE_LENGTH,embeddings_initializer = tf.keras.initializers.Constant(EMBEDDING_MATRIX)))
encoder.add(LSTM(units = 256))
#encoder.add(tf.compat.v1.keras.layers.CuDNNLSTM(units=256))

# Create tensors for Context and Utterance
context_input = Input(shape=(MAX_SENTENCE_LENGTH,),dtype='float32')
utterance_input = Input(shape=(MAX_SENTENCE_LENGTH,),dtype='float32')

# Encode Context and Utterance through LSTM
encoded_context = encoder(context_input)            # Shape = (None,256)
encoded_utterance = encoder(utterance_input)        # Actual response encoding (None,256) --> Need to take its transpose to make dimenions add up

concatenate = tf.math.multiply(encoded_context,encoded_utterance)

#middle_layer = Dense((256),activation='sigmoid')(concatenate)
probability = Dense((1),activation='sigmoid')(concatenate)

dual_encoder = Model(inputs=[context_input,utterance_input],outputs = probability)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.001, epsilon=1e-07, centered=False,name='RMSprop',clipnorm = True)
dual_encoder.compile(loss='binary_crossentropy',optimizer=optimizer)
print(dual_encoder.summary())
#print("Trainable variables :",dual_encoder.trainable_weights[4])
"""https://stackoverflow.com/questions/55413421/importerror-failed-to-import-pydot-please-install-pydot-for-example-with"""
plot_model(dual_encoder, os.path.join(OUTPUT_PATH,'dual_encoder.png'),show_shapes = True)
print(dual_encoder.summary())

#print(dual_encoder.trainable_weights[3])

print("Now loading UDC data...")
input_dir = OUTPUT_PATH + '\\'
train_c, train_r, train_l = pickle.load(open(input_dir + 'train.pkl', 'rb'))
test_c, test_r, test_l = pickle.load(open(input_dir + 'test.pkl', 'rb'))
dev_c, dev_r, dev_l = pickle.load(open(input_dir + 'dev.pkl', 'rb'))
train_l = np.asarray(train_l)
test_l = np.asarray(test_l)
dev_l = np.asarray(dev_l)
print('Found %s training samples.' % len(train_c))
print('Found %s dev samples.' % len(dev_c))
print('Found %s test samples.' % len(test_c))

print("Training the model...")
histories = Histories(([dev_c, dev_r], dev_l))
  
bestAcc = 0.0
patience = 0 
epochs = 1
for ep in range(epochs):
  dual_encoder.fit([train_c,train_r],train_l,batch_size = 128,callbacks=[histories], verbose=1,epochs=1)
  
  curAcc =  histories.accs[0]
  if curAcc >= bestAcc:
     bestAcc = curAcc
     patience = 0
  else:
     patience = patience + 1
  #classify the test set
  y_pred = dual_encoder.predict([test_c, test_r],batch_size=16)          
  
  #print("Perform on test set after Epoch: " + str(ep) + "...!")    
  recall_k = compute_recall_ks(y_pred[:,0])
  
  #stop training the model when patience = 10
  if patience > 10:
     #print("Early stopping at epoch: "+ str(ep))
     break
if True:
      print("Now saving the model... at {}".format(OUTPUT_PATH + '/' + 'dual_encoder_classifier_model.h5'))
      dual_encoder.save(OUTPUT_PATH + '/' + 'dual_encoder_classifier_model.h5')
