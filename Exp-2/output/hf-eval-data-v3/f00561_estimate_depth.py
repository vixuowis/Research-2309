# function_import --------------------

from transformers import pipeline
import numpy as np

# function_code --------------------

def estimate_depth(image):
    """
    This function estimates the depth of an image using a pre-trained model from Hugging Face Transformers.

    Args:
        image (np.array): The input image for which the depth needs to be estimated.

    Returns:
        np.array: The depth map of the input image.
    """
    depth_model = pipeline('depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode')
    depth_map = depth_model(image)
    return depth_map

# test_function_code --------------------

def test_estimate_depth():
    """
    This function tests the 'estimate_depth' function by using a random image.
    """
    sample_image = np.random.rand(100, 100, 3)
    depth_map = estimate_depth(sample_image)
    assert isinstance(depth_map, np.ndarray), 'The output should be a numpy array.'
    assert depth_map.shape == sample_image.shape, 'The output shape should match the input shape.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_estimate_depth()