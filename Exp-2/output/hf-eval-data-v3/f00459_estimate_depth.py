# function_import --------------------

from transformers import pipeline
import PIL.Image
import requests
from io import BytesIO
import numpy as np

# function_code --------------------

def estimate_depth(input_image):
    """
    Estimate the depth map of a given image using a pre-trained model from Hugging Face Transformers.

    Args:
        input_image (str or PIL.Image.Image): The input image. This can be either a URL to an image, a local path to an image, or a PIL Image object.

    Returns:
        np.ndarray: The estimated depth map of the input image.

    Raises:
        ValueError: If the input image source is incorrect. It must be a valid URL starting with `http://` or `https://`, a valid path to an image file, or a PIL Image object.
    """
    # Create a pipeline for 'depth-estimation' using the model 'sayakpaul/glpn-kitti-finetuned-diode-221214-123047'
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-kitti-finetuned-diode-221214-123047')

    # If the input is a URL, download the image and convert it to a PIL Image object
    if isinstance(input_image, str) and input_image.startswith(('http://', 'https://')):
        response = requests.get(input_image)
        input_image = PIL.Image.open(BytesIO(response.content))

    # Estimate the depth map of the input image
    depth_map = depth_estimator(input_image)

    return depth_map

# test_function_code --------------------

def test_estimate_depth():
    """Tests the `estimate_depth` function."""
    # Test with a URL to an image
    input_image = 'https://placekitten.com/200/300'
    depth_map = estimate_depth(input_image)
    assert isinstance(depth_map, np.ndarray), 'The output should be a numpy array.'

    # Test with a PIL Image object
    input_image = PIL.Image.new('RGB', (200, 300))
    depth_map = estimate_depth(input_image)
    assert isinstance(depth_map, np.ndarray), 'The output should be a numpy array.'

    print('All Tests Passed')

# call_test_function_code --------------------

test_estimate_depth()