# requirements_file --------------------

!pip install -U transformers Pillow requests torch

# function_import --------------------

from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests

# function_code --------------------

def estimate_image_depth(image_url: str) -> Image:
    """Estimate the depth map of an image using DPT model.

    Args:
        image_url (str): The URL of the image to be processed.

    Returns:
        Image: The depth image constructed from the estimated depth map.

    Raises:
        ValueError: If the image cannot be retrieved from the URL.
    """
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError('Image cannot be retrieved from the URL') from e

    processor = DPTImageProcessor.from_pretrained('Intel/dpt-large')
    model = DPTForDepthEstimation.from_pretrained('Intel/dpt-large')
    inputs = processor(images=image, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(predicted_depth.unsqueeze(1), size=image.size[::-1], mode='bicubic', align_corners=False)
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype('uint8')

    depth_image = Image.fromarray(formatted)
    return depth_image

# test_function_code --------------------

def test_estimate_image_depth():
    print('Testing started.')
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    print('Testing case [1/1] started.')
    depth_image = estimate_image_depth(url)
    assert depth_image is not None, 'Test case [1/1] failed: depth_image is None'
    print('Testing case [1/1] finished.')
    print('Testing finished.')

# call_test_function_line --------------------

test_estimate_image_depth()