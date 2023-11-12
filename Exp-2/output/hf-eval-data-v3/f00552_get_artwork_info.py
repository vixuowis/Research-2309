# function_import --------------------

from PIL import Image
import requests
from transformers import BlipProcessor, Blip2ForConditionalGeneration

# function_code --------------------

def get_artwork_info(image_path: str, question: str) -> str:
    """
    Get information about an artwork based on an image and a question.

    Args:
        image_path (str): The path to the image of the artwork.
        question (str): The question about the artwork.

    Returns:
        str: The answer to the question based on the image of the artwork.
    """
    processor = BlipProcessor.from_pretrained('Salesforce/blip2-flan-t5-xl')
    model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-flan-t5-xl')

    raw_image = Image.open(image_path).convert('RGB')

    inputs = processor(raw_image, question, return_tensors='pt')
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)

    return answer

# test_function_code --------------------

def test_get_artwork_info():
    """
    Test the function get_artwork_info.
    """
    image_path = 'https://placekitten.com/200/300'
    question = 'What is the historical background of this artwork?'
    answer = get_artwork_info(image_path, question)
    assert isinstance(answer, str), 'The answer should be a string.'

    question = 'Who is the artist of this artwork?'
    answer = get_artwork_info(image_path, question)
    assert isinstance(answer, str), 'The answer should be a string.'

    question = 'What is the style of this artwork?'
    answer = get_artwork_info(image_path, question)
    assert isinstance(answer, str), 'The answer should be a string.'

    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_get_artwork_info())