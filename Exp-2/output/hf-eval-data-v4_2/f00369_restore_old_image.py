# requirements_file --------------------

!pip install -U huggingface_hub Pillow tensorflow numpy

# function_import --------------------

from huggingface_hub import from_pretrained_keras
from PIL import Image
import tensorflow as tf
import numpy as np

# function_code --------------------

def restore_old_image(image_path: str) -> Image.Image:
    """
    Restores an old, blurry image using a pre-trained deblurring model.

    Args:
        image_path: A string representing the file path to the blurry image.

    Returns:
        A PIL Image object of the restored, deblurred image.

    Raises:
        FileNotFoundError: If the image_path does not exist.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image not found at {image_path}')
    
    # Load and process the image
    image = Image.open(image_path)
    image = np.array(image)
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (256, 256))
    
    # Load the pre-trained model
    model = from_pretrained_keras('google/maxim-s3-deblurring-gopro')
    
    # Deblur the image
    predictions = model.predict(tf.expand_dims(image, 0))
    deblurred_image = tf.squeeze(predictions, axis=0)
    deblurred_image = tf.clip_by_value(deblurred_image, 0, 255)
    deblurred_image = tf.cast(deblurred_image, tf.uint8)
    
    deblurred_image = Image.fromarray(deblurred_image.numpy())
    return deblurred_image

# test_function_code --------------------

def test_restore_old_image():
    print('Testing started.')
    # Use a sample image path for testing
    sample_image_path = 'sample_blurry_image.jpg'
    
    # Test case 1: Check if the function raises FileNotFoundError for a non-existent file
    print('Testing case [1/3] started.')
    non_existent_path = 'non_existent_image.jpg'
    try:
        restore_old_image(non_existent_path)
        assert False, f'Test case [1/3] failed: FileNotFoundError was not raised for a non-existent file path.'
    except FileNotFoundError:
        assert True
    
    # Test case 2: Check if the restored image is a PIL Image object
    print('Testing case [2/3] started.')
    restored_image = restore_old_image(sample_image_path)
    assert isinstance(restored_image, Image.Image), f'Test case [2/3] failed: The returned object is not a PIL Image.'
    
    # Test case 3: Verify that the returned image has the required dimensions (256, 256)
    print('Testing case [3/3] started.')
    width, height = restored_image.size
    assert (width, height) == (256, 256), f'Test case [3/3] failed: The returned image does not have the dimensions 256x256.'
    print('Testing finished.')

# call_test_function_line --------------------

test_restore_old_image()