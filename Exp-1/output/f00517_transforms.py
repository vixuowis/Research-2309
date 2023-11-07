from typing import *
import tensorflow as tf

def transforms(image):
    # Transpose the image to channels-first layout
    # Return the transformed image
    image = tf.keras.utils.img_to_array(image)
    image = tf.transpose(image, (2, 0, 1))
    return image
