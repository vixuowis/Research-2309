# function_import --------------------

from huggingface_hub import from_pretrained_keras
from PIL import Image
import tensorflow as tf
import numpy as np

# function_code --------------------

def deblur_image(image_path):
    """
    This function deblurs an image using the 'google/maxim-s3-deblurring-gopro' model from Keras.

    Args:
        image_path (str): The path to the image to be deblurred.

    Returns:
        deblurred_image (PIL.Image.Image): The deblurred image.
    """
    image = Image.open(image_path)
    image = np.array(image)
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (256, 256))

    model = from_pretrained_keras('google/maxim-s3-deblurring-gopro')
    predictions = model.predict(tf.expand_dims(image, 0))

    deblurred_image = tf.squeeze(predictions, axis=0)
    deblurred_image = tf.clip_by_value(deblurred_image, 0, 255)
    deblurred_image = tf.cast(deblurred_image, tf.uint8)

    deblurred_image = Image.fromarray(deblurred_image.numpy())
    return deblurred_image

# test_function_code --------------------

def test_deblur_image():
    """
    This function tests the deblur_image function by deblurring a sample image and checking the output type.
    """
    deblurred_image = deblur_image('path/to/sample_image.jpg')
    assert isinstance(deblurred_image, Image.Image), 'The output should be a PIL Image.'

# call_test_function_code --------------------

test_deblur_image()