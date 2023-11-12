# function_import --------------------

from transformers import AutoModel
import numpy as np

# function_code --------------------

def estimate_depth(input_image_path):
    """
    Estimate the depth of a scene in an image using a pretrained model.

    Args:
        input_image_path (str): The path to the input image.

    Returns:
        np.array: The estimated depth map of the input image.

    Raises:
        FileNotFoundError: If the input image file does not exist.
    """
    # Load the pretrained model
    depth_estimator = AutoModel.from_pretrained('sayakpaul/glpn-nyu-finetuned-diode-221215-095508')

    # Preprocess the input image
    processed_image = preprocess_image(input_image_path)

    # Estimate the depth map
    predicted_depth_map = depth_estimator(processed_image)

    return predicted_depth_map

# test_function_code --------------------

def test_estimate_depth():
    """
    Test the estimate_depth function.
    """
    # Test case 1: Normal case
    input_image_path = 'test_images/test_image1.jpg'
    expected_result = np.array([...])
    assert np.allclose(estimate_depth(input_image_path), expected_result, rtol=1e-05, atol=1e-08)

    # Test case 2: The input image file does not exist
    try:
        estimate_depth('non_existent_file.jpg')
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('The FileNotFoundError was not raised')

    print('All Tests Passed')

# call_test_function_code --------------------

test_estimate_depth()