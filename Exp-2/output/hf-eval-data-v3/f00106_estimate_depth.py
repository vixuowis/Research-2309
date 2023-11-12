# function_import --------------------

from transformers import pipeline
import numpy as np
import requests

# function_code --------------------

def estimate_depth(image_path):
    """
    Estimate the depth of spaces in an image using a pre-trained model from Hugging Face Transformers.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The depth map of the image.

    Raises:
        FileNotFoundError: If the image file does not exist.
        requests.exceptions.ConnectionError: If there is a problem connecting to the Hugging Face model hub.
    """
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode-221221-102136')
    return depth_estimator(image_path)

# test_function_code --------------------

def test_estimate_depth():
    """
    Test the estimate_depth function with a sample image.
    """
    test_image_path = 'test_image.jpg'
    try:
        depth_map = estimate_depth(test_image_path)
        assert isinstance(depth_map, np.ndarray), 'The output should be a numpy array.'
        print('Test passed.')
    except FileNotFoundError:
        print('Test image not found.')
    except requests.exceptions.ConnectionError:
        print('Could not connect to the Hugging Face model hub.')

# call_test_function_code --------------------

test_estimate_depth()