# function_import --------------------

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# function_code --------------------

def generate_image_description(img_url):
    '''
    Generate a description of an image using the BlipForConditionalGeneration model.

    Args:
        img_url (str): The URL or local path of the image to be described.

    Returns:
        str: The generated description of the image.
    '''
    
    # Check if the input is a valid URL
    try: 
        requests.get(img_url) 
    except Exception as e:
        print('The given img_url is not a valid URL!')
        return None
  
    # Load image from url and preprocess it for BLIP
    processor = BlipProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    if 'http' in img_url:  # If the input is a URL
        img = Image.open(requests.get(img_url, stream=True).raw)
        
    else:                   # If the input is a path to local image
        with open(img_url,"rb") as f:
            img = Image.open(f)
            
    model_inputs = processor(images=[img], return_tensors="pt", padding='max_length', max_length=72, truncation=True, add_special_tokens=True, return_attention_mask=True)
    
    # Load the BLIP model and generate a description of the image.
    blip = BlipForConditionalGeneration.from_pretrained('openai/blip-base-400m') 
    generated_ids = blip.generate(model_inputs['input_ids'], num_beams=3, max_length=60, min_length=5)
    
    return processor.batch_decode(generated_ids)[0]

# test_function_code --------------------

def test_generate_image_description():
    '''
    Test the function generate_image_description.
    '''
    img_url1 = 'https://placekitten.com/200/300'
    img_url2 = 'https://placekitten.com/400/500'
    img_url3 = 'https://placekitten.com/600/700'
    assert isinstance(generate_image_description(img_url1), str)
    assert isinstance(generate_image_description(img_url2), str)
    assert isinstance(generate_image_description(img_url3), str)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_image_description()