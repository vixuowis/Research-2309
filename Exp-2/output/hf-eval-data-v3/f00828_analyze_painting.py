# function_import --------------------

from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image
import requests

# function_code --------------------

def analyze_painting(img_url: str, question: str) -> str:
    '''
    Analyze a painting and answer a question about it.

    Args:
        img_url (str): The URL of the painting image.
        question (str): The question to be answered about the painting.

    Returns:
        str: The answer to the question.

    Raises:
        Exception: If there is an error in processing the image or generating the answer.
    '''
    try:
        processor = BlipProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
        model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        inputs = processor(raw_image, question, return_tensors='pt')
        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        raise Exception('Error in analyzing painting: ' + str(e))

# test_function_code --------------------

def test_analyze_painting():
    '''
    Test the analyze_painting function.
    '''
    img_url = 'https://placekitten.com/200/300'
    question = 'What colors are predominant in this painting?'
    answer = analyze_painting(img_url, question)
    assert isinstance(answer, str), 'The answer should be a string.'
    assert len(answer) > 0, 'The answer should not be empty.'

    img_url = 'https://placekitten.com/400/600'
    question = 'What is the style of this painting?'
    answer = analyze_painting(img_url, question)
    assert isinstance(answer, str), 'The answer should be a string.'
    assert len(answer) > 0, 'The answer should not be empty.'

    img_url = 'https://placekitten.com/800/1200'
    question = 'Who is the artist of this painting?'
    answer = analyze_painting(img_url, question)
    assert isinstance(answer, str), 'The answer should be a string.'
    assert len(answer) > 0, 'The answer should not be empty.'

    print('All Tests Passed')

# call_test_function_code --------------------

test_analyze_painting()