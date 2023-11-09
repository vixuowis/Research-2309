# function_import --------------------

from transformers import pipeline

# function_code --------------------

def estimate_depth(input_image):
    """
    Estimate the depth map of an input image using a pre-trained model.

    Args:
        input_image (str or PIL Image): The input image for which to estimate the depth. This can be either a path to an image file or a PIL Image object.

    Returns:
        ndarray: The estimated depth map of the input image.
    """
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-kitti-finetuned-diode-221214-123047')
    depth_map = depth_estimator(input_image)
    return depth_map

# test_function_code --------------------

def test_estimate_depth():
    """
    Test the estimate_depth function.
    """
    # Use a sample image for testing
    input_image = 'path/to/sample/image'
    depth_map = estimate_depth(input_image)
    assert depth_map is not None, 'The depth map should not be None.'
    assert depth_map.shape[0] > 0, 'The depth map should have a positive height.'
    assert depth_map.shape[1] > 0, 'The depth map should have a positive width.'

# call_test_function_code --------------------

test_estimate_depth()