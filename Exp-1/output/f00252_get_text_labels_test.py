from f00252_get_text_labels import *
import tensorflow as tf

id2label = {0: 'O', 1: 'B-location', 2: 'I-location', 3: 'B-group'}
logits = tf.constant([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])

predicted_token_class = get_text_labels(logits, id2label)
print(predicted_token_class)
# Output: ['O', 'I-location']
