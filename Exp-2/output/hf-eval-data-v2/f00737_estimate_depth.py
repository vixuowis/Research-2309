# function_import --------------------

from transformers import AutoModel
from PIL import Image
import torch

# function_code --------------------

def estimate_depth(image_path):
    """
    This function estimates the depth of elements in architectural designs from 2D images of the structures.
    It uses a pre-trained model from Hugging Face Transformers.

    Args:
        image_path (str): The path to the 2D image of the architectural design.

    Returns:
        depth_pred (torch.Tensor): The estimated depth of the elements in the image.
    """
    model = AutoModel.from_pretrained('sayakpaul/glpn-nyu-finetuned-diode-221116-104421')
    image = Image.open(image_path)
    tensor_image = torch.tensor(image).unsqueeze(0)  # convert image to tensor
    depth_pred = model(tensor_image)  # estimate depth of elements in the image
    return depth_pred

# test_function_code --------------------

def test_estimate_depth():
    """
    This function tests the estimate_depth function.
    It uses a sample image and checks if the output is a torch.Tensor.
    """
    sample_image_path = 'sample_image.jpg'  # replace with path to a sample image
    depth_pred = estimate_depth(sample_image_path)
    assert isinstance(depth_pred, torch.Tensor), 'The output should be a torch.Tensor.'

# call_test_function_code --------------------

test_estimate_depth()