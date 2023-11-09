# function_import --------------------

from transformers import AutoModel
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch

# function_code --------------------

def estimate_depth(image):
    """
    This function estimates the depth of the environment using a monocular image.
    It uses a pre-trained model from Hugging Face Transformers.

    Args:
        image (PIL.Image): The input monocular image.

    Returns:
        torch.Tensor: The estimated depth map.
    """
    # Initialize the model
    model = AutoModel.from_pretrained('sayakpaul/glpn-nyu-finetuned-diode-221122-082237')

    # Preprocess input image
    transforms = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image = transforms(image).unsqueeze(0)

    # Compute depth map
    with torch.no_grad():
        depth_map = model(input_image)

    return depth_map

# test_function_code --------------------

def test_estimate_depth():
    """
    This function tests the estimate_depth function.
    It uses a sample image from an online source.
    """
    # Load a sample image from an online source
    image = Image.open(requests.get('https://example.com/sample.jpg', stream=True).raw)

    # Estimate the depth
    depth_map = estimate_depth(image)

    # Check the output type
    assert isinstance(depth_map, torch.Tensor), 'Output type should be torch.Tensor'

    # Check the output shape
    assert len(depth_map.shape) == 3, 'Output shape should be 3D (C, H, W)'

    # Check the output values
    assert torch.all(depth_map >= 0), 'All depth values should be non-negative'

# call_test_function_code --------------------

test_estimate_depth()