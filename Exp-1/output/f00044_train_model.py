from typing import *
from tensorflow.keras.optimizers import Adam

def train_model(model, tf_dataset):
    model.compile(optimizer=Adam(3e-5))
    model.fit(tf_dataset)
