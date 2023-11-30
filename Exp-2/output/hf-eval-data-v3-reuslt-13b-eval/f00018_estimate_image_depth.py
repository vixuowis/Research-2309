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

    try:
        # Load the image and convert it to NumPy format:
        response = requests.get(image_url)
        image = Image.open(response.content).convert('RGB')
        image = np.asarray(image)

        # Create a processor instance that will perform all of the necessary preprocessing steps and batching for the model:
        try:
            processor = DPTImageProcessor.from_pretrained("IntelPicasso/dpt-large")
            processor.default_image_size = 384
            #print('Using large model')
        except Exception as e:
            print(e)
            processor = DPTImageProcessor.from_pretrained("IntelPicasso/dpt-small")
            processor.default_image_size = 384
            #print('Could not find large model, using small instead')
        input_height = input_width = processor.default_image_size

        # Create a PyTorch tensor with the properly processed image:
        img_tensor = processor(image, return_tensors="pt")["pixel_values"][0]
        batch_img_tensor = torch.unsqueeze(img_tensor, 0)

        # Load the pretrained model from Hugging Face:
        try:
            model = DPTForDepthEstimation.from_pretrained("IntelPicasso/dpt-large")
            input_height = input_width = processor.default_image_size
            #print('Using large model')
        except Exception as e:
            print(e)
            model = DPTForDepthEstimation.from_pretrained("IntelPicasso/dpt-small")
            input_height = input_width = processor.default_image_size
            #print('Could not find large model, using small instead')
        model.eval()

        # Run the image through the network and convert it to NumPy array:
        with torch.no_grad():
            outputs = model(batch_img_tensor)

        depth_estimation = outputs[0]["pred_depth"].numpy

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