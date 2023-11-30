# function_import --------------------

from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests

# function_code --------------------

def estimate_image_depth(image_url):
    '''
    Estimate the depth of an image using a pretrained model from Hugging Face Transformers.

    Args:
        image_url (str): The URL of the image to be processed.

    Returns:
        depth (PIL.Image): The depth estimation of the image.

    Raises:
        requests.exceptions.RequestException: If the image cannot be loaded from the provided URL.
        RuntimeError: If there is a problem loading the pretrained model.
    '''

    # Load Image --------------------

    try:
        response = requests.get(image_url)
        img = Image.open(response.content).convert("RGB")
    except Exception as e:
        raise ValueError('Image could not be loaded from the provided URL')

    # Preprocess --------------------
        
    image_processor = DPTImageProcessor()
    
    input, original_size = image_processor(img)
    input = input.unsqueeze(0).to("cuda")
    model = DPTForDepthEstimation(pretrained=True)
    model.to('cuda')
    model.eval()
    
    # Process --------------------
    
    with torch.no_grad():
        try:
            outputs = model(input)
            predictions = (outputs[("depth", 0)]).view(*original_size).detach().cpu().numpy()
            
        except RuntimeError as e:
            raise ValueError('Pretrained depth estimation model could not be loaded')
        
    # Postprocess --------------------
    
    depth = (np.tanh(predictions) + 1) / 2 * 255
    depth = Image.fromarray((depth*255).astype('uint8'))
    
    return depth

# test_function_code --------------------

def test_estimate_image_depth():
    '''
    Test the estimate_image_depth function with different test cases.
    '''
    test_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    result = estimate_image_depth(test_image_url)
    assert isinstance(result, Image.Image), 'The result should be a PIL Image.'
    test_image_url = 'https://placekitten.com/200/300'
    result = estimate_image_depth(test_image_url)
    assert isinstance(result, Image.Image), 'The result should be a PIL Image.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_estimate_image_depth()