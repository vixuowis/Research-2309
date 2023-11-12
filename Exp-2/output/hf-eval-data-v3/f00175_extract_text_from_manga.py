# function_import --------------------

from transformers import pipeline
import requests
from PIL import Image
from io import BytesIO

# function_code --------------------

def extract_text_from_manga(image_url: str) -> str:
    """
    Extracts text from a manga image using the Hugging Face Transformers OCR pipeline.

    Args:
        image_url (str): The URL of the manga image.

    Returns:
        str: The extracted text from the manga image.

    Raises:
        KeyError: If the OCR task is not recognized by the pipeline.
    """
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    ocr_pipeline = pipeline('ocr', model='kha-white/manga-ocr-base')
    manga_text = ocr_pipeline(image)
    return manga_text

# test_function_code --------------------

def test_extract_text_from_manga():
    """
    Tests the extract_text_from_manga function.
    """
    test_image = 'https://placekitten.com/200/300'
    result = extract_text_from_manga(test_image)
    assert isinstance(result, str), 'The result should be a string.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_extract_text_from_manga()