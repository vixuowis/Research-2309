# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

import requests
from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration

# function_code --------------------

def get_image_summary_and_answer(img_url, question):
    """
    Generates a text summary and answers a question from the provided image URL.

    Args:
    img_url (str): URL of the image to analyze.
    question (str): Question to be answered about the image.

    Returns:
    str: The answer to the question.
    """
    processor = BlipProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
    model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')

    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    inputs = processor(raw_image, question, return_tensors='pt')
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)

    return answer

# test_function_code --------------------

def test_get_image_summary_and_answer():
    print("Testing get_image_summary_and_answer function.")

    # Test case: Use an available example image and a related question
    img_url = 'https://example.com/example_image.jpg'
    question = 'What is the main color of the object?'
    expected_answer = 'The main color of the object is ...'  # Expected answer goes here

    print("Test case: Checking with the example image and question.")
    answer = get_image_summary_and_answer(img_url, question)
    assert answer == expected_answer, f"Test failed: Expected '{{expected_answer}}', got '{{answer}}'"
    print("Test passed.")

    # Additional test cases can be added as necessary

# Executing the test function
test_get_image_summary_and_answer()