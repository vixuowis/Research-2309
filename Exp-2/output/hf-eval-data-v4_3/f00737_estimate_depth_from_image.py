# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch", "Pillow"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModel
from PIL import Image
import torch

# function_code --------------------

def estimate_depth_from_image(image_path):
    """
    Estimates the depth of elements in an architectural design from a 2D image.

    Args:
        image_path (str): The path to the 2D image file of the architectural design.

    Returns:
        torch.Tensor: A tensor representing the estimated depth map.

    Raises:
        FileNotFoundError: If the image file does not exist at the provided path.
    """
    model = AutoModel.from_pretrained('sayakpaul/glpn-nyu-finetuned-diode-221116-104421')
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f'The image file does not exist: {image_path}')
    tensor_image = torch.tensor(image).unsqueeze(0)  # convert image to tensor
    depth_pred = model(tensor_image)  # estimate depth of elements in the image
    return depth_pred

# test_function_code --------------------

def test_estimate_depth_from_image():
    print("Testing started.")
    # Assuming we have an available sample image in the test directory
    sample_image_path = 'test_images/sample_architectural_design.jpg'

    # Test case 1: Valid image file path
    print("Testing case [1/1] started.")
    try:
        depth_map = estimate_depth_from_image(sample_image_path)
        assert isinstance(depth_map, torch.Tensor), "Depth map is not a tensor."
    except Exception as e:
        assert False, f"Test case [1/1] failed: {str(e)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_estimate_depth_from_image()