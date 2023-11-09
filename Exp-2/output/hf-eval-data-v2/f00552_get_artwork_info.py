# function_import --------------------

from PIL import Image
import requests
from transformers import BlipProcessor, Blip2ForConditionalGeneration

# function_code --------------------

def get_artwork_info(image_path: str, question: str) -> str:
    """
    This function takes an image path and a question as input, and returns a string answer.
    It uses the Blip2ForConditionalGeneration model from Hugging Face Transformers to generate the answer.

    Args:
        image_path (str): The path to the image file.
        question (str): The question about the artwork.

    Returns:
        str: The answer to the question.
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
    This function tests the get_artwork_info function.
    It uses a sample image and question, and checks if the returned answer is a string.
    """
    image_path = 'path/to/sample/image.jpg'
    question = 'What is the historical background of this artwork?'
    answer = get_artwork_info(image_path, question)

    assert isinstance(answer, str), 'The function should return a string.'

# call_test_function_code --------------------

test_get_artwork_info()