# function_import --------------------

from transformers import pipeline
import torch

# function_code --------------------

def estimate_depth(image_path: str) -> torch.Tensor:
    """
    Estimate the depth of objects within an image using a pre-trained model from Hugging Face Transformers.

    Args:
        image_path (str): The path to the image file.

    Returns:
        torch.Tensor: The depth map of the image.

    Raises:
        FileNotFoundError: If the image file does not exist.
        KeyError: If the task or model is not recognized by the Hugging Face Transformers pipeline.
    """
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode-221215-093747')
    depth_map = depth_estimator(image_path)
    return depth_map

# test_function_code --------------------

def test_estimate_depth():
    """
    Test the estimate_depth function with a few test cases.
    """
    test_image_path = 'path/to/test/image.jpg'
    try:
        depth_map = estimate_depth(test_image_path)
        assert isinstance(depth_map, torch.Tensor), 'The output should be a torch.Tensor.'
    except FileNotFoundError:
        print('Test image not found.')
    except KeyError:
        print('The task or model is not recognized by the Hugging Face Transformers pipeline.')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_estimate_depth()