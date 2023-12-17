# requirements_file --------------------

!pip install -U transformers>=4.24.0,torch>=1.12.1,Pillow,torchvision

# function_import --------------------

from transformers import AutoModel
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize

# function_code --------------------

def estimate_depth(image_path):
    """
    Estimate the depth map of an input image using a pre-trained depth estimation model.

    Parameters:
        image_path (str): The file path of the input image.

    Returns:
        torch.Tensor: The estimated depth map as a 2D tensor.
    """
    # Load the pre-trained model
    model = AutoModel.from_pretrained('sayakpaul/glpn-nyu-finetuned-diode-221122-082237')

    # Define image transformations
    transformations = Compose([
        Resize((640, 480)),
        ToTensor()
    ])

    # Load and transform the input image
    input_image = Image.open(image_path)
    input_image = transformations(input_image).unsqueeze(0)  # Add batch dimension

    # Make sure the model is in evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        depth_map = model(input_image)

    return depth_map


# test_function_code --------------------

def test_estimate_depth():
    print("Testing estimate_depth function.")

    # Test with a sample image
    sample_image_path = 'sample_image.jpg'  # Replace with path to a real image

    # Expected output format check
    depth_map = estimate_depth(sample_image_path)
    assert isinstance(depth_map, torch.Tensor), "The output is not a torch.Tensor"

    # Check the output shape
    expected_shape = (1, 1, 640, 480)  # Assuming the output depth map has the shape (B, C, H, W)
    assert depth_map.shape == expected_shape, f"Output shape {depth_map.shape} does not match expected shape {expected_shape}"

    print("estimate_depth function passed all tests.")

# Run the test function
test_estimate_depth()
