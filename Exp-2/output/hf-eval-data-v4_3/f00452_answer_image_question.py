# requirements_file --------------------

import subprocess

requirements = ["transformers", "pillow", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image
import requests

# function_code --------------------

def answer_image_question(img_url: str, question: str) -> str:
    """
    Answer a question related to an image using the Salesforce BLIP-2 model.

    Args:
        img_url: A string representing the URL of the image.
        question: A string representing the question related to the image.

    Returns:
        A string containing the answer to the question.

    Raises:
        ValueError: If the URL or question is invalid or if the image cannot be processed.
    """
    try:
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    except Exception as e:
        raise ValueError('Failed to load and process the image.') from e

    processor = BlipProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
    model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
    inputs = processor(raw_image, question, return_tensors='pt')
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)

    return answer

# test_function_code --------------------

def test_answer_image_question():
    print("Testing started.")
    # Test case 1: Valid image URL and question
    print("Testing case [1/2] started.")
    valid_img_url = 'https://example.com/image_of_your_pet_dogs.jpg'
    question_1 = 'What breed are the dogs in the picture?'
    assert answer_image_question(valid_img_url, question_1), 'Test case [1/2] failed: expected a non-empty answer.'

    # Test case 2: Invalid image URL
    print("Testing case [2/2] started.")
    invalid_img_url = 'https://example.com/non_existent_image.jpg'
    question_2 = 'What breed are the dogs in the picture?'
    try:
        answer_image_question(invalid_img_url, question_2)
        assert False, 'Test case [2/2] failed: expected a ValueError.'
    except ValueError:
        assert True
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_image_question()