# function_import --------------------

from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image
import requests

# function_code --------------------

def get_ingredients_info(img_url: str, question: str = 'What are the ingredients of this dish?') -> str:
    """
    This function takes an image URL and a question as input, processes the image using BlipProcessor, generates a textual
    description using Blip2ForConditionalGeneration, and returns the decoded output.

    Args:
        img_url (str): The URL of the image to be processed.
        question (str, optional): The question to be answered by the model. Defaults to 'What are the ingredients of this dish?'.

    Returns:
        str: The decoded output from the model, providing a textual description of the image.
    """
    processor = BlipProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
    model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    inputs = processor(raw_image, question, return_tensors='pt')
    out = model.generate(**inputs)
    ingredient_info = processor.decode(out[0], skip_special_tokens=True)
    return ingredient_info

# test_function_code --------------------

def test_get_ingredients_info():
    """
    This function tests the get_ingredients_info function by providing a sample image URL and a question.
    The output is printed.
    """
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    question = 'What are the ingredients of this dish?'
    print(get_ingredients_info(img_url, question))

# call_test_function_code --------------------

test_get_ingredients_info()