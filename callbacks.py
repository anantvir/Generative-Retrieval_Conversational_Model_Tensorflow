import numpy as np
import tensorflow as tf
import os
from helpers import compute_recall_ks,str2bool,recall

DIRNAME_ABSOLUTE = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(DIRNAME_ABSOLUTE, 'output')
class Histories(tf.keras.callbacks.Callback):
    def __init__(self,val_data):
      super().__init__()
      self.validation_data = val_data
    def on_train_begin(self, logs={}): 
        self.accs = []
        self.losses = []
        self.f = open(f"{OUTPUT_PATH}/training_logs.txt",'a')

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        print(self.validation_data[1].shape)
        y_pred = self.model.predict(self.validation_data[0],batch_size = 16)
        
        recall_k = compute_recall_ks(y_pred[:,0])
        
        self.accs.append(recall_k[10][1]) # append the recall 1@10 
        
        self.f.write("Loss for epoch : {} is {}".format(str(epoch),str(logs.get('loss')) + '\n'))
        return