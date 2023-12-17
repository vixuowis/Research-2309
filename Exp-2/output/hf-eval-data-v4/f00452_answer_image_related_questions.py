# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image
import requests

# function_code --------------------

def answer_image_related_questions(img_url, question):
    # Load the image from the provided URL
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    
    # Initialize the processor and model
    processor = BlipProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
    model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
    
    # Process the image and question
    inputs = processor(raw_image, question, return_tensors='pt')
    
    # Generate the answer to the question
    output = model.generate(**inputs)
    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer

# test_function_code --------------------

def test_answer_image_related_questions():
    print("Testing started.")

    # Example URL and question
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    question = 'How many dogs are in the picture?'

    # Expected answer (this will depend on the actual content of the image)
    expected_answer = 'Two dogs'

    # Get the answer from the function
    answer = answer_image_related_questions(img_url, question)

    # Check if the answer is correct
    assert answer.lower() == expected_answer.lower(), f"Test failed: Expected '{expected_answer}', but got '{answer}'"
    print("Testing finished.")