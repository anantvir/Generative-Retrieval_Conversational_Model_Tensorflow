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
from preprocessing import read_train_TFRecords
from custom_layer import CustomLayer
import pickle


"""-------------------------------------- Constants ------------------------------------------"""
DIRNAME_ABSOLUTE = os.path.dirname(__file__)
GLOVE_EMBEDDING_PATH = os.path.join(DIRNAME_ABSOLUTE, 'glove')
OUTPUT_PATH = os.path.join(DIRNAME_ABSOLUTE, 'output')
MAX_SENTENCE_LENGTH = 160
MAX_NB_WORDS = 10000
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
num_words = min(MAX_NB_WORDS,len(word_index)) + 1       # +1 for <UNK>
EMBEDDING_MATRIX = np.zeros((num_words,EMBEDDING_DIM))

for word,index in word_index.items():
	if index > MAX_NB_WORDS:
		continue
	embedding_vector = embeddings_index.get(word)       #.get() used because for key not found it returns None and not KeyError !
	if embedding_vector is not None:
		EMBEDDING_MATRIX[index] = embedding_vector

print("Building Dual Encoder LSTM ...")

embedder = Sequential()
embedder.add(Embedding(input_dim = num_words,output_dim = EMBEDDING_DIM,input_length = MAX_SENTENCE_LENGTH,embeddings_initializer = tf.keras.initializers.Constant(EMBEDDING_MATRIX)))

encoder = Sequential()
encoder.add(Embedding(input_dim = num_words,output_dim = EMBEDDING_DIM,input_length = MAX_SENTENCE_LENGTH,embeddings_initializer = tf.keras.initializers.Constant(EMBEDDING_MATRIX)))
encoder.add(LSTM(units = 256))

# Create tensors for Context and Utterance
context_input = Input(shape=(MAX_SENTENCE_LENGTH,),dtype='float32')
utterance_input = Input(shape=(MAX_SENTENCE_LENGTH,),dtype='float32')

# Encode Context and Utterance through LSTM
encoded_context = encoder(context_input)            # Shape = (None,256)
encoded_utterance = encoder(utterance_input)        # Actual response encoding (None,256) --> Need to take its transpose to make dimenions add up

"""Use Custom layer to make GradientTape work"""
custom_layer = CustomLayer(256,256)
generated_response = custom_layer(encoded_context)

projection = tf.linalg.matmul(generated_response,tf.transpose(encoded_utterance))
probability = tf.math.sigmoid(projection)

dual_encoder = Model(inputs=[context_input,utterance_input],outputs = probability)
#print("Trainable variables :",dual_encoder.trainable_weights)
"""https://stackoverflow.com/questions/55413421/importerror-failed-to-import-pydot-please-install-pydot-for-example-with"""
plot_model(dual_encoder, os.path.join(OUTPUT_PATH,'dual_encoder.png'),show_shapes = True)


#dual_encoder.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop',metrics=['accuracy'])
# print("Summary of Dual Encoder LSTM :",dual_encoder.summary())
# def create_batched_dataset(data_path):
# 	tfrecord_dataset = tf.data.TFRecordDataset(os.path.join(data_path,"train.tfrecords"))
# 	parsed_dataset = tfrecord_dataset.map(read_train_TFRecords,num_parallel_calls = 8)
# 	parsed_dataset = parsed_dataset.repeat()
# 	parsed_dataset = parsed_dataset.shuffle(SHUFFLE_BUFFER)
# 	parsed_dataset = parsed_dataset.batch(BATCH_SIZE)
# 	return parsed_dataset

#parsed_dataset = create_batched_dataset(OUTPUT_PATH)

# reference - https://www.tensorflow.org/guide/keras/train_and_evaluate
optimizer = RMSprop(learning_rate=0.001, rho=0.9, momentum=0.1, epsilon=1e-07, centered=False)

input_dir = 'D:\\Courses\\Chatbot\\blog\\'
train_c, train_r, train_l = pickle.load(open(input_dir + 'outputtrain.pkl', 'rb'))
test_c, test_r, test_l = pickle.load(open(input_dir + 'outputtest.pkl', 'rb'))
dev_c, dev_r, dev_l = pickle.load(open(input_dir + 'outputdev.pkl', 'rb'))
train_l = np.asarray(train_l)
test_l = np.asarray(test_l)
dev_l = np.asarray(dev_l)
print('Found %s training samples.' % len(train_c))
print('Found %s dev samples.' % len(dev_c))
print('Found %s test samples.' % len(test_c))

print("Now training the model...")

epochs = 10
for epoch in range(epochs):
	print('Start of epoch %d' % (epoch,))

  # Iterate over the batches of the dataset.
	for step,row in enumerate(zip(train_c,train_r,train_l)):
		input_batch_context,input_batch_utterance,input_batch_label = row
		input_batch_context = np.reshape(input_batch_context,((BATCH_SIZE,160)))
		input_batch_utterance = np.reshape(input_batch_utterance,(BATCH_SIZE,160))
		#print("Context :",input_batch_context)
		with tf.GradientTape() as tape:

			# Run the forward pass of the layer. The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
			#emb = embedder(input_batch_context)
			#print(emb)
			#encoded_context = encoder(input_batch_context)
			#print(encoded_context)
			pred = dual_encoder([input_batch_context,input_batch_utterance],training = True)
			#print("Prediction :",pred)
			#print("Label :",input_batch_label)
			# Compute the loss value for this minibatch.
			loss_value = binary_crossentropy(input_batch_label, pred)
			#print("Loss :",loss_value)

		# Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
		grads = tape.gradient(loss_value, dual_encoder.trainable_weights)
		#print(grads)
		# Run one step of gradient descent by updating the value of the variables to minimize the loss.
		optimizer.apply_gradients(zip(grads, dual_encoder.trainable_weights))

		# Log every 200 batches.
		if step % 200 == 0:
			print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
			print('Seen so far: %s samples' % ((step + 1) * BATCH_SIZE))





