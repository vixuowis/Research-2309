from typing import *
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

def train_model(tf_train_dataset, tf_eval_dataset, num_epochs, callbacks):
    
    # Train the model
    model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=num_epochs, callbacks=callbacks)
