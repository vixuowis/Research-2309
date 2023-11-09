from huggingface_hub import from_pretrained_keras
from PIL import Image
import tensorflow as tf
import numpy as np


def deblur_image(image_path):
    """
    This function takes the path of a blurry image as input and returns a deblurred image.
    It uses the 'google/maxim-s3-deblurring-gopro' model from Keras to perform the deblurring.
    """
    # Load the old, blurry image using Image.open and convert it to a numpy array
    image = Image.open(image_path)
    image = np.array(image)
    
    # Convert the numpy array to a TensorFlow tensor, and resize it to 256x256 pixels as required by the model
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (256, 256))
    
    # Load the 'google/maxim-s3-deblurring-gopro' model, which is designed for image deblurring tasks
    model = from_pretrained_keras('google/maxim-s3-deblurring-gopro')
    
    # Make a prediction using the model to deblur the image
    predictions = model.predict(tf.expand_dims(image, 0))
    
    # Process the output to get the deblurred image
    deblurred_image = tf.squeeze(predictions, axis=0)
    deblurred_image = tf.clip_by_value(deblurred_image, 0, 255)
    deblurred_image = tf.cast(deblurred_image, tf.uint8)
    
    # Convert the tensor back to an image and save it
    deblurred_image = Image.fromarray(deblurred_image.numpy())
    deblurred_image.save('deblurred_image.png')
    
    return deblurred_image