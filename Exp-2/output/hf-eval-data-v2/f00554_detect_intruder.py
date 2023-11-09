# function_import --------------------

from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import requests

# function_code --------------------

def detect_intruder(image_path: str, question: str = 'Who entered the room?') -> str:
    """
    This function uses a pretrained model from Hugging Face Transformers to answer questions based on an image.
    The model is specialized in multimodal visual question answering.

    Args:
        image_path (str): The path to the image file.
        question (str, optional): The question to be answered. Defaults to 'Who entered the room?'.

    Returns:
        str: The answer to the question.
    """
    processor = BlipProcessor.from_pretrained('Salesforce/blip-vqa-capfilt-large')
    model = BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-capfilt-large')

    cctv_image = Image.open(image_path)

    inputs = processor(cctv_image, question, return_tensors='pt')
    answer = model.generate(**inputs)
    return processor.decode(answer[0], skip_special_tokens=True)

# test_function_code --------------------

def test_detect_intruder():
    """
    This function tests the detect_intruder function.
    It uses a sample image and question, and checks if the output is a string.
    """
    image_path = 'test_image.jpg'
    question = 'Who is in the picture?'
    answer = detect_intruder(image_path, question)
    assert isinstance(answer, str), 'The function should return a string.'

# call_test_function_code --------------------

test_detect_intruder()