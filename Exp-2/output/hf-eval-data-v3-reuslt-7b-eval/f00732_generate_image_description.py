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
    
    tokenizer = BlipTokenizer() # from transformers

    processor = BlipProcessor.from_pretrained('blip/supervised_transformer')

    model = BlipForConditionalGeneration.from_pretrained("blip/supervised_transformer")
    
    img = Image.open(requests.get(img_url, stream=True).raw)  # or Image.open('path/to/img')
    image_inputs = processor(images=img, return_tensors="pt", padding=True)  
    
    input_ids = torch.tensor([tokenizer.encode("")])
    
    outputs = model.generate(
        input_ids, 
        attention_mask=None, 
        decoder_input_ids=(torch.ones((1, 4), device=model.device) * tokenizer._convert_token_to_id('<s>')).long(),
        num_beams=5,
        num_return_sequences=20,
        max_length=30,
        **image_inputs, 
    )
    
    description = tokenizer.decode(outputs[0])[:-1] # remove eos token
    return description

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