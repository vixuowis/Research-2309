# function_import --------------------

from transformers import AutoModel
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
from PIL import Image
import urllib.request
import io

# function_code --------------------

def estimate_depth(image):
    """
    Estimate the depth of the environment using a monocular image.

    Args:
        image (PIL.Image): The input image.

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
    Test the estimate_depth function.
    """
    from PIL import Image
    import urllib.request
    import io

    # Test case 1: Online image
    url = 'https://placekitten.com/200/300'
    with urllib.request.urlopen(url) as url:
        f = io.BytesIO(url.read())
    img = Image.open(f)
    depth_map = estimate_depth(img)
    assert isinstance(depth_map, torch.Tensor), 'Output is not a torch.Tensor'

    # Test case 2: Local image
    img = Image.open('test_image.jpg')
    depth_map = estimate_depth(img)
    assert isinstance(depth_map, torch.Tensor), 'Output is not a torch.Tensor'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_estimate_depth()