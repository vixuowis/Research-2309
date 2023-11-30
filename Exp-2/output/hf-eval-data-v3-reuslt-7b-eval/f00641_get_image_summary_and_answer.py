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
        # Load Image
        img = requests.get(img_url).content
        img = Image.open(BytesIO(img))
        
        # Summarize Image
        processor = BlipProcessor.from_pretrained('blip/blip-large-uncased')
        model = Blip2ForConditionalGeneration.from_pretrained('blip/blip-large-uncased')
        
        inputs = processor(img, return_tensors="pt")
        summary = model.generate(**inputs)
        summary = processor.batch_decode(summary, skip_special_tokens=True)[0]

        # Generate answer
        tokenizer = model.blip.encoder 
        input_ids = tokenizer(question + ' [SEP] ' + summary, return_tensors="pt").input_ids
        outputs = model.generate(input_ids)
        
        return processor.decode(outputs[0], skip_special_tokens=True)[1:-1]
    except Exception as e:
        raise Exception('Error in processing the image or generating an answer')

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