# function_import --------------------

from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

# function_code --------------------

def get_image_answer(url: str, question: str) -> str:
    """
    This function takes an image URL and a question as input, and returns the answer to the question based on the image.
    It uses the ViLT model fine-tuned on VQAv2 from Hugging Face Transformers.

    Args:
        url (str): The URL of the image.
        question (str): The question to be answered.

    Returns:
        str: The answer to the question.

    Raises:
        OSError: If there is not enough disk space to download the model.
    """    
    
    # Load model from Hugging Face Hub
    processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-huf')
    vilt = ViltForQuestionAnswering.from_pretrained('dandelin/vilt-b32-huf', return_dict=True)
    
    # Download image to local storage
    response = requests.get(url, stream=True)
    with open("temp.jpg", 'wb') as img:
        for chunk in response.iter_content():
            if chunk:  # filter out keep-alive new chunks
                img.write(chunk)
                
    # Load image into Pillow Image object
    pil_image = Image.open("temp.jpg")
    
    inputs = processor(
        text=question,
        images=pil_image,
        return_tensors="pt",
        padding='max_length',
        max_length=256,
        truncation=True)
    
    # Run inference on model and obtain answer token
    outputs = vilt(**inputs)
    answer_token = processor.convert_tokens_to_string([int(outputs["answer"])])
        
    return answer_token


# test_function_code --------------------

def test_get_image_answer():
    """
    This function tests the get_image_answer function with a few test cases.
    """
    assert isinstance(get_image_answer('http://images.cocodataset.org/val2017/000000039769.jpg', 'How many people are in this photo?'), str)
    assert isinstance(get_image_answer('https://placekitten.com/200/300', 'What is in this photo?'), str)
    assert isinstance(get_image_answer('https://placekitten.com/200/300', 'Is there a cat in this photo?'), str)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_get_image_answer()