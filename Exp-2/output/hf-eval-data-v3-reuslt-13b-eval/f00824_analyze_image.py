# function_import --------------------

import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests

# function_code --------------------

def analyze_image(url):
    '''
    Analyze an image from a given URL using the Vision Transformer (ViT) model.

    Args:
        url (str): The URL of the image to be analyzed.

    Returns:
        last_hidden_states (torch.Tensor): The last hidden states from the ViT model.

    Raises:
        OSError: If there is a problem with the network connection or the image file.
    '''

    # get the input data into appropriate format
    feature_extractor = ViTImageProcessor(size=256, crop_size=256) 
    
    try:
        # download and read in the image
        image_data = requests.get(url).content
        
        with Image.open(image_data) as img:
            imgs = [img]
            
            inputs = feature_extractor(imgs, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state[0].tolist() # [1, 512, 768]

        return last_hidden_state
    except:
        raise OSError("Problem with network or image file.") from None
    

# test_function_code --------------------

def test_analyze_image():
    '''
    Test the analyze_image function with different test cases.
    '''
    url1 = 'https://placekitten.com/200/300'
    url2 = 'https://placekitten.com/400/600'
    url3 = 'https://placekitten.com/800/1200'
    assert analyze_image(url1).shape == torch.Size([1, 197, 768])
    assert analyze_image(url2).shape == torch.Size([1, 197, 768])
    assert analyze_image(url3).shape == torch.Size([1, 197, 768])
    return 'All Tests Passed'


# call_test_function_code --------------------

test_analyze_image()