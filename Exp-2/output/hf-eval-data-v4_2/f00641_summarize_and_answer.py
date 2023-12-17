# requirements_file --------------------

!pip install -U requests Pillow transformers

# function_import --------------------

import requests
from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration

# function_code --------------------

def summarize_and_answer(img_url, question):
    """Generate a summary and answer a question based on the content of an image.
    
    Args:
        img_url (str): The URL of the image to analyze.
        question (str): The question to answer based on the image.

    Returns:
        tuple: A tuple containing the text summary and answer to the question.

    Raises:
        ValueError: If the image is not accessible or other errors occur.
    """
    try:
        processor = BlipProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
        model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        inputs = processor(raw_image, question, return_tensors='pt')
        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)

        return (summary, answer)
    except Exception as e:
        raise ValueError('Error processing the image or generating an answer: ' + str(e))

# test_function_code --------------------

def test_summarize_and_answer():
    print("Testing started.")
    
    # Test case 1: Testing with a valid image URL and question
    print("Testing case [1/1] started.")
    summary, answer = summarize_and_answer('https://example.com/test_image.jpg', 'What is shown in the image?')
    assert type(summary) == str and type(answer) == str, f"Test case [1/1] failed: Expected string outputs, got {type(summary)} and {type(answer)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_and_answer()