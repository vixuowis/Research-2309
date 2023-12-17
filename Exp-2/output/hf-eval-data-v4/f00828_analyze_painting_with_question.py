# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image
import requests

# function_code --------------------


def analyze_painting_with_question(img_url: str, question: str) -> str:
    """
    Analyzes an image of a painting using the specified BLIP-2 model and answers
    questions related to the painting.

    Parameters:
    img_url (str): URL of the painting image.
    question (str): A question about the painting.

    Returns:
    str: The answer to the given question about the painting.
    """
    processor = BlipProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
    model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    inputs = processor(raw_image, question, return_tensors='pt')
    outputs = model.generate(**inputs)
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    return answer

# test_function_code --------------------


def test_analyze_painting_with_question():
    print("Testing started.")
    test_img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    test_question_1 = 'What is depicted in the painting?'
    test_question_2 = 'What colors are predominant in this painting?'
    test_question_3 = 'What is the artistic style of this painting?'

    # Test case 1
    print("Testing case [1/3] started.")
    answer_1 = analyze_painting_with_question(test_img_url, test_question_1)
    assert isinstance(answer_1, str), f"Test case [1/3] failed: The result should be a string, got {type(answer_1)}"

    # Test case 2
    print("Testing case [2/3] started.")
    answer_2 = analyze_painting_with_question(test_img_url, test_question_2)
    assert isinstance(answer_2, str), f"Test case [2/3] failed: The result should be a string, got {type(answer_2)}"

    # Test case 3
    print("Testing case [3/3] started.")
    answer_3 = analyze_painting_with_question(test_img_url, test_question_3)
    assert isinstance(answer_3, str), f"Test case [3/3] failed: The result should be a string, got {type(answer_3)}"
    print("Testing finished.")
