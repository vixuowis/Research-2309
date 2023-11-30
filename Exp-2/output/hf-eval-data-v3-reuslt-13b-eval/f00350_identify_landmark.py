# function_import --------------------

from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration
import requests

# function_code --------------------

def identify_landmark(img_url: str, question: str) -> str:
    '''
    Identify the landmark in the image and answer the question about the landmark.

    Args:
        img_url (str): The URL of the image of the landmark.
        question (str): The question to be answered by the model based on the image.

    Returns:
        str: The answer or information about the landmark.
    '''
    url = 'https://storage.googleapis.com/sweep-data/blip/models/landmarks/model_100k/model_100k.pt'
    
    # Load BLIP model and processor
    model = Blip2ForConditionalGeneration.from_pretrained(url)
    processor = BlipProcessor.from_pretrained("models/blip_marblc")
    
    # Get Image
    img = requests.get(img_url).content
    img = Image.open(BytesIO(img))

    # Preprocess image for the model
    inputs = processor([img], return_tensors="pt", padding='max_length', max_length=2048, truncation=True)
    outputs = model.generate(inputs['pixel_values'], num_beams=5, early_stopping=True, max_length=150)
    
    # Decode output from tokenized format to string
    answer = processor.decode(outputs[0])
    
    return answer

# test_function_code --------------------

def test_identify_landmark():
    '''
    Test the identify_landmark function.
    '''
    img_url = 'https://placekitten.com/200/300'
    question = 'What is the name of this landmark?'
    answer = identify_landmark(img_url, question)
    assert isinstance(answer, str), 'The answer should be a string.'
    assert len(answer) > 0, 'The answer should not be an empty string.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_identify_landmark()