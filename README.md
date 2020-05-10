# Generative and Retrieval based Conversational Models in Tensorflow
Generative and Retrieval Chatbots in Tensorflow using TFX, TF Transform, Apache Beam

Retrieval --> Dual LSTM Encoder Siamese Model

Generative --> Seq2seq Model

# Dual Encoder LSTM as described in the publication
![image](https://user-images.githubusercontent.com/27782859/81504334-946b8900-92b6-11ea-9c43-c25caec62727.png)


# Below is a plot of our model from Tensorflow
![dual_encoder](https://user-images.githubusercontent.com/27782859/81504261-4d7d9380-92b6-11ea-94b6-c462d1542e50.png)


# Hyperparams
Embedding Size : 300

Hidden Size : 256

Batch Size : 256

Epochs : 50

Optimizer : RMSProp

Learning rate : 0.001

Embeddings : Glove 300d 


Reference: 
1) https://arxiv.org/pdf/1506.08909.pdf
2) http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/
3) https://github.com/tensorflow/transform/blob/599691c8b94bbd6ee7f67c11542e7fef1792a566/examples/sentiment_example.py#L80
4) https://www.tensorflow.org/tfx/transform/get_started
5) https://towardsdatascience.com/working-with-tfrecords-and-tf-train-example-36d111b3ff4d
6) https://basmaboussaha.wordpress.com/2017/10/18/implementation-of-dual-encoder-using-keras/comment-page-1/?unapproved=229&moderation-hash=f6485213fcee44e23ad52e4c5c231424#comment-229
7) Original Implementation in Theano https://github.com/npow/ubottu
