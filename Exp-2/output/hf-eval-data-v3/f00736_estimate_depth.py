# function_import --------------------

from transformers import pipeline
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# function_code --------------------

def estimate_depth(image_url: str) -> np.ndarray:
    """
    Estimate the depth of objects in an image captured by a drone's camera.

    Args:
        image_url (str): The URL of the image to estimate depth.

    Returns:
        np.ndarray: The depth map of the image.
    """
    # Create a depth estimation pipeline using the 'glpn-nyu-finetuned-diode' model
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode')

    # Load the image from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Estimate the depth of the image
    depth_map = depth_estimator(image)

    return depth_map

# test_function_code --------------------

def test_estimate_depth():
    """
    Test the estimate_depth function.
    """
    # Test with a random image
    image_url = 'https://placekitten.com/200/300'
    depth_map = estimate_depth(image_url)
    assert isinstance(depth_map, np.ndarray), 'The depth map should be a numpy array.'

    # Test with another random image
    image_url = 'https://placekitten.com/400/600'
    depth_map = estimate_depth(image_url)
    assert isinstance(depth_map, np.ndarray), 'The depth map should be a numpy array.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_estimate_depth()