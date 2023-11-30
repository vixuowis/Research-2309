# function_import --------------------

from transformers import DPTForDepthEstimation
import numpy as np

# function_code --------------------

def estimate_depth(model_name: str, drone_footage: np.ndarray) -> np.ndarray:
    """
    Estimate the depth in drone footage using a pre-trained DPTForDepthEstimation model.

    Args:
        model_name (str): The name of the pre-trained DPTForDepthEstimation model.
        drone_footage (np.ndarray): The drone footage to estimate depth from.

    Returns:
        np.ndarray: The estimated depth map.
    """
    
    # Get the pre-trained DPTForDepthEstimation model with pre-trained weights for the given model name.
    model = DPTForDepthEstimation.from_pretrained(model_name)
    
    # Estimate depth from drone footage.
    estimated_depth = model(drone_footage)["pred_depth"].cpu().detach().numpy()[0][0]  # (1, 1, 640, 384) -> (640, 384)
    
    return estimated_depth


# test_function_code --------------------

def test_estimate_depth():
    """
    Test the estimate_depth function.
    """
    model_name = 'hf-tiny-model-private/tiny-random-DPTForDepthEstimation'
    drone_footage = np.random.rand(100, 100, 3)
    depth_map = estimate_depth(model_name, drone_footage)
    assert depth_map.shape == drone_footage.shape, 'The shape of the depth map should be the same as the drone footage.'
    assert np.all(depth_map == 0), 'The depth map should be all zeros for this placeholder implementation.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_estimate_depth()