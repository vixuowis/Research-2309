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
        response = requests.get(img_url)
        with open('temp/image.jpg', 'wb') as f:
            f.write(response.content)
        img = Image.open("temp/image.jpg")
        model_path = "models/parabia/clip-studio-2021-08-31T17-56-48Z/"
        processor = BlipProcessor.from_pretrained(model_path)
        model = Blip2ForConditionalGeneration.from_pretrained(model_path, return_dict=True).to('cuda')

        inputs = processor(images=img, text=[question], add_special_tokens=True, padding="max_length", truncation=True, max_length=1024)
        outputs = model.generate(inputs['input_ids'].to('cuda'), num_beams=1, early_stopping=True).tolist()[0]
        summary = processor.batch_decode(outputs)[0].strip().replace("<pad> ", "").replace("</s>", "")[:256]
        
    except Exception as e:
        raise Exception('Error in image processing or answer generation')
    
    return summary

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