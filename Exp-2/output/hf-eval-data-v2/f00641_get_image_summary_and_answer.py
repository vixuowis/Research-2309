# function_import --------------------

import requests
from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration

# function_code --------------------

def get_image_summary_and_answer(img_url: str, question: str) -> str:
    """
    This function uses the 'Blip2ForConditionalGeneration' model from Hugging Face Transformers to generate a text summary and answer a question from an image.

    Args:
        img_url (str): The URL of the image.
        question (str): The question to be answered.

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
    """
    This function tests the 'get_image_summary_and_answer' function.
    """
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    question = 'how many dogs are in the picture?'
    answer = get_image_summary_and_answer(img_url, question)
    assert isinstance(answer, str), 'The result should be a string.'

# call_test_function_code --------------------

test_get_image_summary_and_answer()