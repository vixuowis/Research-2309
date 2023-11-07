from typing import *
import tensorflow as tf

def get_text_labels(logits, id2label):
    predicted_token_class_ids = tf.math.argmax(logits, axis=-1)
    predicted_token_class = [id2label[t] for t in predicted_token_class_ids[0].numpy().tolist()]
    return predicted_token_class
