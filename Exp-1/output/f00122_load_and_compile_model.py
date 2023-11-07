from typing import *
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam


def load_and_compile_model(model_name, num_labels):
    # Load the pre-trained model
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Set the optimizer
    optimizer = Adam(learning_rate=1e-5)
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
    
    return model
