# function_import --------------------

import torch
from transformers import AutoModel
from torchvision.io import read_image

# function_code --------------------

def estimate_depth(image_path: str) -> torch.Tensor:
    """
    Estimate the depth of an image using a pre-trained model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        torch.Tensor: The estimated depth of the image.
    """
    image_input = read_image(image_path)
    depth_estimator = AutoModel.from_pretrained('sayakpaul/glpn-nyu-finetuned-diode-221228-072509')
    predicted_depth = depth_estimator(image_input.unsqueeze(0))
    return predicted_depth

# test_function_code --------------------

def test_estimate_depth():
    """
    Test the estimate_depth function.
    """
    # Test with a sample image
    depth = estimate_depth('sample_image.jpg')
    assert isinstance(depth, torch.Tensor), 'The result should be a torch.Tensor'
    assert depth.shape == (1, 1, 480, 640), 'The shape of the result is incorrect'
    print('All Tests Passed')

# call_test_function_code --------------------

test_estimate_depth()