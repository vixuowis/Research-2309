# requirements_file --------------------

import subprocess

requirements = ["transformers", "pillow"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import requests

# function_code --------------------

def detect_intruder(image_path, question):
    """
    Detects and answers a question about an intruder based on CCTV image input.

    Args:
        image_path (str): The path to the CCTV image.
        question (str): The question being asked about the image.

    Returns:
        str: The answer generated by the AI model.

    Raises:
        FileNotFoundError: If the image_path does not refer to an existing file.
        ValueError: If the model fails to generate an answer.
    """
    processor = BlipProcessor.from_pretrained('Salesforce/blip-vqa-capfilt-large')
    model = BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-capfilt-large')

    try:
        cctv_image = Image.open(image_path)
    except IOError:
        raise FileNotFoundError(f'Image file not found at {image_path}')

    inputs = processor(cctv_image, question, return_tensors='pt')
    answer = model.generate(**inputs)
    decoded_answer = processor.decode(answer[0], skip_special_tokens=True)

    if not decoded_answer:
        raise ValueError('Model failed to generate an answer')

    return decoded_answer

# test_function_code --------------------

def test_detect_intruder():
    print("Testing started.")

    # Test case 1: Valid image path and question
    print("Testing case [1/3] started.")
    image_path = 'path/to/valid/image.jpg'
    question = 'Who entered the room?'
    answer = detect_intruder(image_path, question)
    assert answer, f"Test case [1/3] failed: Expected a valid answer, got {answer}"

    # Test case 2: Image not found
    print("Testing case [2/3] started.")
    invalid_image_path = 'path/to/non-existent/image.jpg'
    try:
        detect_intruder(invalid_image_path, question)
        assert False, "Test case [2/3] failed: Expected FileNotFoundError"
    except FileNotFoundError:
        assert True

    # Test case 3: Model fails to generate an answer
    print("Testing case [3/3] started.")
    question = 'Invalid question that causes model to fail'
    try:
        detect_intruder(image_path, question)
        assert False, "Test case [3/3] failed: Expected ValueError"
    except ValueError:
        assert True
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_intruder()