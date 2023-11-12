# function_import --------------------

from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image
import requests

# function_code --------------------

def get_answer_from_image(img_url: str, question: str) -> str:
    """
    This function takes an image URL and a question as input, and returns the answer to the question based on the image.
    It uses the Blip2ForConditionalGeneration model from Hugging Face Transformers.

    Args:
        img_url (str): The URL of the image.
        question (str): The question related to the image.

    Returns:
        str: The answer to the question based on the image.
    """
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    processor = BlipProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
    model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
    inputs = processor(raw_image, question, return_tensors='pt')
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

# test_function_code --------------------

def test_get_answer_from_image():
    """
    This function tests the get_answer_from_image function.
    It uses a few test cases with known answers.
    """
    assert get_answer_from_image('https://placekitten.com/200/300', 'What color is the cat?') == 'The cat is brown.'
    assert get_answer_from_image('https://placekitten.com/200/300', 'Is the cat sitting or standing?') == 'The cat is sitting.'
    assert get_answer_from_image('https://placekitten.com/200/300', 'Is the cat looking at the camera?') == 'Yes, the cat is looking at the camera.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_answer_from_image()