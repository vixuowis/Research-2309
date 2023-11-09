# function_import --------------------

import torch
from transformers import AutoModel

# function_code --------------------

def estimate_pool_depth(underwater_photo):
    """
    Estimate the depth of a pool using a pre-trained model from Hugging Face Transformers.

    Args:
        underwater_photo (PIL.Image): An underwater photo of the pool.

    Returns:
        depth_estimation (torch.Tensor): The estimated depth of the pool.
    """
    model = AutoModel.from_pretrained('hf-tiny-model-private/tiny-random-GLPNForDepthEstimation')

    # Pre-process underwater photo and convert to tensor
    underwater_photo_tensor = preprocess_underwater_photo(underwater_photo)

    # Get depth estimation from the model
    depth_estimation = model(underwater_photo_tensor)

    return depth_estimation

# test_function_code --------------------

def test_estimate_pool_depth():
    """
    Test the function estimate_pool_depth.
    """
    # Load a test underwater photo
    test_photo = load_test_photo()

    # Estimate the depth of the pool
    depth_estimation = estimate_pool_depth(test_photo)

    # Check the type of the output
    assert isinstance(depth_estimation, torch.Tensor), 'The output should be a torch.Tensor.'

    # Check the size of the output
    assert depth_estimation.size(0) > 0, 'The output tensor should not be empty.'

# call_test_function_code --------------------

test_estimate_pool_depth()