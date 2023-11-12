# function_import --------------------

from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# function_code --------------------

def estimate_parking_depth(image_url: str) -> dict:
    """
    Estimate the depth of a parking spot using a depth estimation model.

    Args:
        image_url (str): The URL of the image of the parking spot.

    Returns:
        dict: The depth estimation of the parking spot.
    """
    # Create a depth estimation model
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode-221122-044810')

    # Load the image from the URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Estimate the depth of the parking spot
    depth_estimate = depth_estimator(img)

    return depth_estimate

# test_function_code --------------------

def test_estimate_parking_depth():
    """
    Test the estimate_parking_depth function.
    """
    # Test with a sample image URL
    image_url = 'https://placekitten.com/200/300'
    depth_estimate = estimate_parking_depth(image_url)
    assert isinstance(depth_estimate, dict), 'The result should be a dictionary.'

    # Test with another sample image URL
    image_url = 'https://placekitten.com/300/400'
    depth_estimate = estimate_parking_depth(image_url)
    assert isinstance(depth_estimate, dict), 'The result should be a dictionary.'

    # Test with a third sample image URL
    image_url = 'https://placekitten.com/400/500'
    depth_estimate = estimate_parking_depth(image_url)
    assert isinstance(depth_estimate, dict), 'The result should be a dictionary.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_estimate_parking_depth()