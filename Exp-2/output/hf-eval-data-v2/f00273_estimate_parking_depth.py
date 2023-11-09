# function_import --------------------

from transformers import pipeline

# function_code --------------------

def estimate_parking_depth(parking_spot_image):
    """
    This function estimates the depth of a parking spot using a pre-trained model from Hugging Face Transformers.

    Args:
        parking_spot_image (PIL.Image): An image of the parking spot.

    Returns:
        depth_estimate_image (np.array): A depth estimation of the parking spot.
    """
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode-221122-044810')
    depth_estimate_image = depth_estimator(parking_spot_image)
    return depth_estimate_image

# test_function_code --------------------

def test_estimate_parking_depth():
    """
    This function tests the estimate_parking_depth function by using a sample image.
    """
    sample_image = Image.open('sample_parking_spot.jpg')
    depth_estimate = estimate_parking_depth(sample_image)
    assert isinstance(depth_estimate, np.ndarray), 'The output should be a numpy array.'
    assert depth_estimate.shape == sample_image.size, 'The output shape should match the input image size.'

# call_test_function_code --------------------

test_estimate_parking_depth()