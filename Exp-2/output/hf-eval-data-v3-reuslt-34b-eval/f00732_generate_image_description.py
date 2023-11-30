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

    # Load the pre-trained model and tokenizer
    model = BlipForConditionalGeneration.from_pretrained("DavidMartin/blip-small")
    processor = BlipProcessor.from_pretrained("DavidMartin/blip-small")

    # Process the image and get the corresponding inputs for the model
    img = Image.open(requests.get(img_url, stream=True).raw)
    inputs = processor(text=[], images=processor(img, return_tensors="pt").pixel_values, return_tensors="pt")
    
    # Generate the description using the model and tokenizer
    output_ids_caption = model.generate(**inputs)
    output_caption = processor.batch_decode(output_ids_caption, skip_special_tokens=True)[0].lower()
    
    return output_caption

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