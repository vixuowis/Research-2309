# requirements_file --------------------

import subprocess

requirements = ["transformers", "PIL", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image
import requests

# function_code --------------------

def generate_answer_to_painting(img_url: str, question: str) -> str:
    """
    Generate an answer to the given question based on the provided painting image URL.

    Args:
        img_url (str): The URL of the painting image.
        question (str): The question asked about the painting.

    Returns:
        str: The answer to the question based on the image analysis.

    Raises:
        ValueError: If the image cannot be opened or processed.
    """
    processor = BlipProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
    model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')

    try:
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    except Exception as e:
        raise ValueError(f'Unable to open or process image from URL {img_url}.') from e
    
    inputs = processor(raw_image, question, return_tensors='pt')
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

# test_function_code --------------------

def test_generate_answer_to_painting():
    print("Testing started.")
    
    # TestCase 1: Valid image URL and question
    print("Testing case [1/3] started.")
    img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    question = "how many dogs are in the picture?"
    answer = generate_answer_to_painting(img_url, question)
    assert answer is not None and isinstance(answer, str), f"Test case [1/3] failed: Expected a string answer, got {answer}."

    # TestCase 2: Invalid image URL
    print("Testing case [2/3] started.")
    img_url = "https://nonexistent-url.com/image.jpg"
    question = "What is depicted in the painting?"
    try:
        generate_answer_to_painting(img_url, question)
        assert False, "Test case [2/3] failed: Expected ValueError for invalid image URL."
    except ValueError:
        pass  # Expected

    # TestCase 3: Empty question
    print("Testing case [3/3] started.")
    img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    question = ""
    answer = generate_answer_to_painting(img_url, question)
    assert answer is not None and isinstance(answer, str), f"Test case [3/3] failed: Expected a string answer even with an empty question, got {answer}."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_answer_to_painting()