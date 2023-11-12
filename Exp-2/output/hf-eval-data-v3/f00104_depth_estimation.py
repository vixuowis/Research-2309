# function_import --------------------

from transformers import AutoModel
import torch

# function_code --------------------

def depth_estimation(model_name: str, image: torch.Tensor):
    """
    Function to estimate depth from an image using a pre-trained model.

    Args:
        model_name (str): The name of the pre-trained model to use for depth estimation.
        image (torch.Tensor): The input image tensor.

    Returns:
        torch.Tensor: The depth estimation tensor.
    """
    model = AutoModel.from_pretrained(model_name)
    depth = model(image)
    return depth

# test_function_code --------------------

def test_depth_estimation():
    """
    Test function for depth_estimation function.
    """
    # Test case 1: Check the output type
    image = torch.rand(1, 3, 224, 224)
    depth = depth_estimation('sayakpaul/glpn-nyu-finetuned-diode-221116-104421', image)
    assert isinstance(depth, torch.Tensor), 'Output type is not correct'

    # Test case 2: Check the output shape
    assert depth.shape == (1, 1, 224, 224), 'Output shape is not correct'

    # Test case 3: Check the output values
    assert torch.all(depth >= 0), 'Output contains negative values'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_depth_estimation()