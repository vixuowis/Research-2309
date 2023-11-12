# function_import --------------------

from huggingface_hub import from_pretrained_keras
from PIL import Image
import tensorflow as tf
import numpy as np

# function_code --------------------

def deblur_image(image_path: str, model_name: str = 'google/maxim-s3-deblurring-gopro') -> Image:
    """
    Deblur an image using a pretrained model from Hugging Face Hub.

    Args:
        image_path (str): Path to the input image.
        model_name (str, optional): Name of the pretrained model. Defaults to 'google/maxim-s3-deblurring-gopro'.

    Returns:
        Image: The deblurred image.
    """
    image = Image.open(image_path)
    image = np.array(image)
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (256, 256))

    model = from_pretrained_keras(model_name)
    predictions = model.predict(tf.expand_dims(image, 0))

    deblurred_image = tf.squeeze(predictions, axis=0)
    deblurred_image = tf.clip_by_value(deblurred_image, 0, 255)
    deblurred_image = tf.cast(deblurred_image, tf.uint8)

    return Image.fromarray(deblurred_image.numpy())

# test_function_code --------------------

def test_deblur_image():
    """
    Test the deblur_image function.
    """
    # Test with a blurry image
    image_path = 'https://github.com/sayakpaul/maxim-tf/raw/main/images/Deblurring/input/1fromGOPR0950.png'
    deblurred_image = deblur_image(image_path)
    assert isinstance(deblurred_image, Image.Image), 'The output should be an instance of PIL.Image.Image'

    # Test with a clear image
    image_path = 'https://placekitten.com/200/300'
    deblurred_image = deblur_image(image_path)
    assert isinstance(deblurred_image, Image.Image), 'The output should be an instance of PIL.Image.Image'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_deblur_image()