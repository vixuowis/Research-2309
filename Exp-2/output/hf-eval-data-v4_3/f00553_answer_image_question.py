# requirements_file --------------------

import subprocess

requirements = ["transformers", "requests", "Pillow"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

# function_code --------------------

def answer_image_question(image_url, question_text):
    """
    Answers a question about an image using a pre-trained ViltForQuestionAnswering model.

    Args:
        image_url (str): URL of the image.
        question_text (str): The question in text form.

    Returns:
        str: The answer to the question about the image.

    Raises:
        ValueError: If the image or text is invalid or not found.
    """
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError("Invalid image URL") from e

    try:
        processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-vqa')
        model = ViltForQuestionAnswering.from_pretrained('dandelin/vilt-b32-finetuned-vqa')
    except Exception as e:
        raise ValueError("Failed to load the ViltForQuestionAnswering model") from e

    encoding = processor(image, question_text, return_tensors='pt')
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    answer = model.config.id2label[idx]
    return answer

# test_function_code --------------------

def test_answer_image_question():
    print("Testing started.")

    # Test case 1: Valid image and question
    print("Testing case [1/1] started.")
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    question_text = 'How many people are in this photo?'
    answer = answer_image_question(image_url, question_text)
    assert answer is not None, f"Test case [1/1] failed: Expected an answer, got {answer}"
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_image_question()