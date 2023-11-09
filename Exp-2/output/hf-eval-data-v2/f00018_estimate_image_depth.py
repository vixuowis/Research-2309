# function_import --------------------

from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests

# function_code --------------------

def estimate_image_depth(image_url):
    """
    This function estimates the depth of an image using a pretrained model from Hugging Face Transformers.
    The model used is 'Intel/dpt-large', which is designed for monocular depth estimation.
    
    Args:
        image_url (str): The URL of the image to be analyzed.
    
    Returns:
        depth (Image): An image object representing the depth estimation of the input image.
    
    Raises:
        Exception: If the image cannot be opened.
    """
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise Exception('Unable to open image. Please check the URL.') from e
    
    processor = DPTImageProcessor.from_pretrained('Intel/dpt-large')
    model = DPTForDepthEstimation.from_pretrained('Intel/dpt-large')
    inputs = processor(images=image, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    prediction = torch.nn.functional.interpolate(predicted_depth.unsqueeze(1), size=image.size[::-1], mode='bicubic', align_corners=False)
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype('uint8')
    depth = Image.fromarray(formatted)
    
    return depth

# test_function_code --------------------

def test_estimate_image_depth():
    """
    This function tests the estimate_image_depth function by using a sample image URL.
    The function asserts that the output is an instance of the Image class.
    """
    sample_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    result = estimate_image_depth(sample_image_url)
    assert isinstance(result, Image), 'The result should be an instance of the Image class.'

# call_test_function_code --------------------

test_estimate_image_depth()