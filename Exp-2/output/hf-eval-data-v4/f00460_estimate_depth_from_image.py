# requirements_file --------------------

!pip install -U torch, torchvision, transformers, Pillow

# function_import --------------------

import torch
from transformers import AutoModel
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

# function_code --------------------

def estimate_depth_from_image(image_path):
    """
    Estimate the depth of the environment from a monocular image.

    Parameters:
    image_path (str): Path to the input image.

    Returns:
    torch.Tensor: A depth map of the input image.
    """
    # Preprocess the image
    transforms = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    input_image = transforms(image).unsqueeze(0)

    # Initialize the GLPN model
    model = AutoModel.from_pretrained('sayakpaul/glpn-nyu-finetuned-diode-221122-082237')
    model.eval()  # Set the model to evaluation mode

    # Estimate the depth
    with torch.no_grad():
        depth_map = model(input_image)

    return depth_map

# test_function_code --------------------

def test_estimate_depth_from_image():
    print("Testing estimate_depth_from_image function started.")

    # Test case: Check if the function returns a torch.Tensor
    print("Testing case [1/1] started.")
    result = estimate_depth_from_image('sample_image.jpg')
    assert isinstance(result, torch.Tensor), f"Test case failed: Result is not a torch.Tensor"

    print("Testing finished.")