# requirements_file --------------------

import subprocess

requirements = ["transformers", "pillow"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration

# function_code --------------------

def describe_artwork(image_path, question):
    """
    Generate a description for artwork using the given question.

    Args:
        image_path (str): The file path to the image of the artwork.
        question (str): The question to ask about the artwork.

    Returns:
        str: The generated answer to the given question.

    Raises:
        FileNotFoundError: If the image file does not exist.
        Exception: If there is an error during processing or generation.
    """
    try:
        # Load the image from the specified path
        raw_image = Image.open(image_path).convert('RGB')
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Image file not found: {image_path}') from e

    try:
        # Initialize the BLIP processor and model
        processor = BlipProcessor.from_pretrained('Salesforce/blip2-flan-t5-xl')
        model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-flan-t5-xl')

        # Process the image and question to create inputs for the model
        inputs = processor(raw_image, question, return_tensors='pt')
        out = model.generate(**inputs)

        # Decode the generated tokens to get the answer
        answer = processor.decode(out[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        raise Exception('Error during generation') from e

# test_function_code --------------------

def test_describe_artwork():
    print('Testing started.')

    # Assuming 'artwork_sample.jpg' exists for the test with a relevant question
    image_path = 'artwork_sample.jpg'
    question = 'Who is the artist of this artwork?'

    print('Testing case [1/1] started.')
    try:
        answer = describe_artwork(image_path, question)
        assert isinstance(answer, str), f'Test case [1/1] failed: Expected a string response, got {type(answer)}'
    except Exception as e:
        assert False, f'Test case [1/1] failed: {e}'

    print('Testing finished.')

# call_test_function_line --------------------

test_describe_artwork()