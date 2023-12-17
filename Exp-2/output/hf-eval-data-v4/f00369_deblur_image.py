# requirements_file --------------------

!pip install -U huggingface_hub PIL tensorflow numpy

# function_import --------------------

from huggingface_hub import from_pretrained_keras
from PIL import Image
import tensorflow as tf
import numpy as np

# function_code --------------------

def deblur_image(image_path):
    """
    Deblur the image at the specified path using the pre-trained MAXIM model.

    Parameters:
        image_path (str): The file path to the blurry image.

    Returns:
        Image: The deblurred image.
    """
    # Load the blurry image
    image = Image.open(image_path)
    image = np.array(image)
    # Convert it to a TensorFlow tensor and resize it
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (256, 256))
    # Load the pre-trained MAXIM deblurring model
    model = from_pretrained_keras('google/maxim-s3-deblurring-gopro')
    # Make a prediction to deblur the image
    predictions = model.predict(tf.expand_dims(image, 0))
    # Process the model output and convert it back to an image
    deblurred_image = tf.squeeze(predictions, axis=0)
    deblurred_image = tf.clip_by_value(deblurred_image, 0, 255)
    deblurred_image = tf.cast(deblurred_image, tf.uint8)
    deblurred_image = Image.fromarray(deblurred_image.numpy())
    return deblurred_image

# test_function_code --------------------

def test_deblur_image():
    print("Testing deblur_image function.")
    # The path to an example blurry image
    example_image_path = 'blurry_image_example.jpg'
    # Expected output shape
    expected_shape = (256, 256, 3)
    # Call the function
    deblurred_image = deblur_image(example_image_path)
    # Convert the deblurred image back to array to check shape
    deblurred_image_array = np.array(deblurred_image)
    # Check if image shape is as expected
    assert deblurred_image_array.shape == expected_shape, f"Test case failed: Expected image shape {expected_shape}, but got {deblurred_image_array.shape}"
    print("Testing finished successfully.")

test_deblur_image()