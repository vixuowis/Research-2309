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
    
    # Retrieve the image from the img_url
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))

    # Create a list with the URL of the image and add it to the 'image' key
    data = {
        'image': [img_url], 
        'question': question
    }

    # Define the model, tokenizer and processor for BLIP
    model = Blip2ForConditionalGeneration.from_pretrained('facebook/blip-vqa')
    model = model.to(device)
    tokenizer = BlipProcessor.from_pretrained("facebook/blip-vqa")
    
    # Process the inputs and get the output of the model
    inputs = tokenizer(data['image'], return_tensors="pt", padding=True).to(device)
    vision_outputs = model.vision_model(**inputs) 
    outputs = model(**inputs, vision_outputs=vision_outputs)
    output_answer = tokenizer.decode(outputs[1].argmax(-1)) # get the most likely answer
    
    return output_answer

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