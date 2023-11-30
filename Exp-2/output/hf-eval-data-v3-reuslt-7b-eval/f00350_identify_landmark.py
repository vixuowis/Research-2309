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
    # Download image using urllib
    img_data = requests.get(img_url)
    with open('img/downloaded-img.jpg', 'wb') as fd:
        for chunk in img_data.iter_content():
            fd.write(chunk)
    
    # Load model
    processor = BlipProcessor.from_pretrained("microsoft/BLIP-base")
    model = Blip2ForConditionalGeneration.from_pretrained("microsoft/BLIP-base", num_labels=1, finetuning_task="image_text_generation").to('cuda')

    # Process image and text using BLIP processor
    encoding = processor(images=Image.open('img/downloaded-img.jpg'), questions=question, padding='max_length', return_tensors="pt")
    
    # Generate predictions
    pred = model.generate(**encoding)
    answer = processor.batch_decode(pred, skip_special_tokens=True)[0]

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