# function_import --------------------

from transformers import pipeline

# function_code --------------------

def estimate_depth(image):
    """
    This function estimates the depth of objects in an image captured by a drone's camera.
    It uses the 'glpn-nyu-finetuned-diode' model from Hugging Face's transformers library.

    Args:
        image (PIL.Image or np.ndarray): The input image. It can be a PIL Image object or a numpy array.

    Returns:
        np.ndarray: The depth map of the input image. The depth map is a 2D array where each pixel represents the estimated distance from the camera.
    """
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode')
    depth_map = depth_estimator(image)
    return depth_map

# test_function_code --------------------

def test_estimate_depth():
    """
    This function tests the 'estimate_depth' function with a sample image.
    It asserts that the output depth map is a 2D array with the same shape as the input image.
    """
    # Load a sample image
    image = np.random.rand(100, 100, 3)
    depth_map = estimate_depth(image)
    # Assert that the output is a 2D array with the same shape as the input image
    assert isinstance(depth_map, np.ndarray)
    assert depth_map.shape == image.shape[:2]

# call_test_function_code --------------------

test_estimate_depth()