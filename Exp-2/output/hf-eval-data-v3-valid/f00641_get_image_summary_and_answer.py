# function_import --------------------

import requests
from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration

# function_code --------------------

def get_image_summary_and_answer(img_url: str, question: str) -> str:
    """
    Get a text summary and answer a question from an image using the 'Blip2ForConditionalGeneration' model.

    Args:
        img_url (str): The URL of the image.
        question (str): The question to be answered.

    Returns:
        str: The answer to the question.

    Raises:
        Exception: If there is an error in processing the image or generating the answer.
    """
    try:
        processor = BlipProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
        model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        inputs = processor(raw_image, question, return_tensors='pt')
        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        raise Exception('Error in getting image summary and answer: ' + str(e))

# test_function_code --------------------

def test_get_image_summary_and_answer():
    """
    Test the function 'get_image_summary_and_answer'.
    """
    try:
        assert get_image_summary_and_answer('https://placekitten.com/200/300', 'What is the main color of the object?') is not None
        assert get_image_summary_and_answer('https://placekitten.com/200/300', 'Is there a cat in the image?') is not None
        assert get_image_summary_and_answer('https://placekitten.com/200/300', 'What is the size of the object?') is not None
        print('All Tests Passed')
    except Exception as e:
        print('Test Failed: ' + str(e))

# call_test_function_code --------------------

test_get_image_summary_and_answer()