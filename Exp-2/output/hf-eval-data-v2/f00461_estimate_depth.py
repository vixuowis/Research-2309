# function_import --------------------

from transformers import AutoModel
from torchvision.io import read_image

# function_code --------------------

def estimate_depth(image_path):
    """
    This function estimates the depth of an image using a pre-trained model from Hugging Face Transformers.
    
    Args:
        image_path (str): The path to the image file.
    
    Returns:
        Tensor: The estimated depth of the image.
    """
    # Load the image
    image_input = read_image(image_path)
    
    # Load the pre-trained model
    depth_estimator = AutoModel.from_pretrained('sayakpaul/glpn-nyu-finetuned-diode-221228-072509')
    
    # Estimate the depth of the image
    predicted_depth = depth_estimator(image_input.unsqueeze(0))
    
    return predicted_depth

# test_function_code --------------------

def test_estimate_depth():
    """
    This function tests the estimate_depth function.
    """
    # Define the path to a test image
    test_image_path = 'test_image.jpg'
    
    # Call the function with the test image
    predicted_depth = estimate_depth(test_image_path)
    
    # Assert that the function returns a tensor
    assert isinstance(predicted_depth, torch.Tensor), 'The function should return a tensor.'

# call_test_function_code --------------------

test_estimate_depth()