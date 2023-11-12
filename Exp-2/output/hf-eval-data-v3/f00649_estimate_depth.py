# function_import --------------------

from transformers import pipeline
import numpy as np

# function_code --------------------

def estimate_depth(image_data):
    """
    Estimate the depth map of a single input image using a pre-trained model.

    Args:
        image_data (PIL.Image or np.ndarray): The input image data.

    Returns:
        np.ndarray: The estimated depth map of the input image.
    """
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode-221122-044810')
    depth_map = depth_estimator(image_data)
    return depth_map

# test_function_code --------------------

def test_estimate_depth():
    """
    Test the estimate_depth function.
    """
    from PIL import Image
    import numpy as np
    import requests
    from io import BytesIO

    # Test with a random image from the internet
    url = 'https://placekitten.com/200/300'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    depth_map = estimate_depth(img)
    assert isinstance(depth_map, np.ndarray), 'The output should be a numpy array.'

    # Test with a numpy array
    img_np = np.array(img)
    depth_map = estimate_depth(img_np)
    assert isinstance(depth_map, np.ndarray), 'The output should be a numpy array.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_estimate_depth()