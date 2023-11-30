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
    
    # Create the image processor for ViT model
    image_processor = ViTImageProcessor()

    # Get the image from URL
    try:
        img = Image.open(requests.get(url, stream=True).raw)
    except OSError as e:
        print('OSError - Could not open image at:', url)
        raise e
    except Exception as e:
        print('Exception - Error with URL:', url)
        raise e
    
    # Create the ViT model
    vit = ViTModel.from_pretrained('google/vit-base-patch16-224')

    # Prepare the image and add batch dimension (BATCH_SIZE = 1)
    prepared_img = image_processor(img, return_tensors='pt').pixel_values.view(-1, 224, 224, 3)

    # Get the last hidden states from ViT model
    with torch.no_grad():
        out = vit(prepared_img)
        last_hidden_states = out.last_hidden_state
    
    return last_hidden_states

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