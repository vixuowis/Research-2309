# function_import --------------------

from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image
import requests

# function_code --------------------

def get_ingredients_info(img_url: str, question: str) -> str:
    """
    Process the food images and give textual information about the items.

    Args:
        img_url (str): The URL of the food image.
        question (str): The question about the food item.

    Returns:
        str: The textual information about the food item.

    Raises:
        Exception: If there is an error in processing the image or generating the text.
    """
    try:
        processor = BlipProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
        model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        inputs = processor(raw_image, question, return_tensors='pt')
        out = model.generate(**inputs)
        ingredient_info = processor.decode(out[0], skip_special_tokens=True)
        return ingredient_info
    except Exception as e:
        raise Exception('Error in getting ingredients info: ' + str(e))

# test_function_code --------------------

def test_get_ingredients_info():
    """
    Test the function get_ingredients_info.
    """
    img_url = 'https://placekitten.com/200/300'
    question = 'What are the ingredients of this dish?'
    try:
        result = get_ingredients_info(img_url, question)
        assert isinstance(result, str)
        print('Test passed.')
    except Exception as e:
        print('Test failed. ' + str(e))

# call_test_function_code --------------------

test_get_ingredients_info()