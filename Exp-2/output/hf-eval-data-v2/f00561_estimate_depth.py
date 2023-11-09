# function_import --------------------

from transformers import pipeline

# function_code --------------------

def estimate_depth(image):
    """
    This function uses a pre-trained model from Hugging Face Transformers to estimate the depth of objects in an image.
    The model used is 'sayakpaul/glpn-nyu-finetuned-diode', which is a fine-tuned version of 'vinvino02/glpn-nyu' on the 'diode-subset' dataset.
    
    Args:
        image (PIL.Image or np.ndarray): The input image for which to estimate depth. The image should be in RGB format.
    
    Returns:
        np.ndarray: A depth map of the input image. The depth map is a 2D array with the same width and height as the input image, where each pixel represents the estimated depth of the corresponding pixel in the input image.
    """
    depth_model = pipeline('depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode')
    depth_map = depth_model(image)
    return depth_map

# test_function_code --------------------

def test_estimate_depth():
    """
    This function tests the 'estimate_depth' function by using a sample image and checking the output.
    The test will pass if the output is a 2D array with the same width and height as the input image.
    """
    sample_image = np.random.rand(100, 100, 3)
    depth_map = estimate_depth(sample_image)
    assert isinstance(depth_map, np.ndarray)
    assert depth_map.shape == sample_image.shape[:2]

# call_test_function_code --------------------

test_estimate_depth()