from time import time
from tensorflow.keras.callbacks import Callback

# keras epoch time class inspired by Keras team's Ben Striner
    
class TimeHistory(Callback):
    def __init__(self):
        self.times = []
        
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time() - self.epoch_time_start)