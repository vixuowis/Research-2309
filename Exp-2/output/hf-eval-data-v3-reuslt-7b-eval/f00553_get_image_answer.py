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
    try:
        processor = ViltProcessor.from_pretrained("dandelin/vilt-VQA")
        
        # Download the image locally and save its path for processing it later on
        if url[:4] == "http": 
            img_data = requests.get(url).content
            with open('image.png', 'wb') as handler:
                handler.write(img_data) 
        
        elif url[-3:] == "jpg" or url[-3:] == "JPG": 
            img = Image.open(url)
            img.save("image.png")
        else: 
            raise IOError("URL is not a valid image file.")
        
        if question[-1] != '?': 
            question += "?"
            
        model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-VQA")
        
        encoding = processor(images="image.png", questions=question, return_tensors="pt")
        
        outputs = model(**encoding) 
        answer_ids = outputs.answer_token_logits.argmax(-1)[0]

    # Remove the image from disk and clean up the directory
    finally:
        os.remove("image.png")
    
    return processor.decode(answer_ids)

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