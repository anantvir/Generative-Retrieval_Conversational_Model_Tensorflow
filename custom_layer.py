from tensorflow.keras import layers
import tensorflow as tf


class CustomLayer(layers.Layer):

  def __init__(self, units=256, input_dim=256):
    super(CustomLayer, self).__init__()
    M_init = tf.random_normal_initializer()
    self.M = tf.Variable(initial_value=M_init(shape=(input_dim, units),dtype='float32'),trainable=True,name = "M")

  def call(self, inputs):
    res = tf.matmul(inputs, self.M) 
    #print(res)
    return res

# x = tf.ones((32, 256))
# custom_layer = CustomLayer(256, 256)
# y = custom_layer(x)
# print(y)
# print("Hello")
