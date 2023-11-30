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
    
    try:
        
        # Get the image data using PIL and Image.open().
        response = requests.get(url, stream=True).raw
        image = Image.open(response)

        # Process the image to be analyzed.
        processor = ViTImageProcessor(image_size=(256, 256), pad_to_multiple_of=32, return_tensors="pt")
        inputs = processor(images=image)
        
        # Load the model and analyze the image using the pretrained ViT model.
        model = ViTModel.from_pretrained("google/vit-base-patch16-224", return_dict=True).to('cuda')
        outputs = model(**inputs.to('cuda'))
        
        # Get the last hidden states for the image from ViTModel output.
        last_hidden_states = outputs.last_hidden_state 
    
    except:
        
        print("Analyze image failed.")
        
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