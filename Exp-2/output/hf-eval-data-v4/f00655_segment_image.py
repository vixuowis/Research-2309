# requirements_file --------------------

!pip install -U transformers, torch, Pillow, torchvision

# function_import --------------------

from transformers import UperNetModel
import torch
from PIL import Image
from torchvision.transforms import ToTensor

# function_code --------------------

def segment_image(image_path):
    """
    Segment the objects in an image using a pre-trained UperNet model with a ConvNeXt backbone.

    :param image_path: str, the path to the image to segment
    :return: torch.Tensor, a semantic segmentation map
    """
    # Load the UperNet model
    model = UperNetModel.from_pretrained('openmmlab/upernet-convnext-small')

    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = ToTensor()(image).unsqueeze(0)  # Add batch dimension

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs['pred_logits'].squeeze(0)  # Remove batch dimension


# test_function_code --------------------

def test_segment_image():
    print("Testing started.")
    # Test with a sample image
    sample_image_path = 'sample_image.jpg'  # Replace with a path to a valid image file

    # Test case 1: Check the output type
    print("Testing case [1/3] started.")
    segmentation_map = segment_image(sample_image_path)
    assert isinstance(segmentation_map, torch.Tensor), f"Test case [1/3] failed: Expected output type torch.Tensor, got {type(segmentation_map)}"

    # Test case 2: Check the number of dimensions of the output
    print("Testing case [2/3] started.")
    assert segmentation_map.ndim == 3, f"Test case [2/3] failed: Expected 3 dimensions, got {segmentation_map.ndim}"

    # Test case 3: Check non-emptiness of the output
    print("Testing case [3/3] started.")
    assert segmentation_map.numel() > 0, f"Test case [3/3] failed: Output is empty"
    print("Testing finished.")

# Run the test function
test_segment_image()
