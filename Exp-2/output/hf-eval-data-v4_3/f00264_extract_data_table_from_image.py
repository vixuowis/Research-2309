# requirements_file --------------------

import subprocess

requirements = ["transformers", "requests", "pillow"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import requests
from PIL import Image

# function_code --------------------

def extract_data_table_from_image(image_path: str, model_name: str = 'google/deplot') -> str:
    """
    Extracts data tables from a plot or chart image using a pre-trained model.

    Args:
        image_path: A string representing the filepath or URL of the image.
        model_name: An optional string representing the pre-trained model to be used for extraction. Default is 'google/deplot'.

    Returns:
        A string representing the linearized table extracted from the image.

    Raises:
        ValueError: If the image cannot be opened.
        RuntimeError: If the model generation fails.
    """
    # Load the pre-trained model and the processor
    model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
    processor = Pix2StructProcessor.from_pretrained(model_name)

    # Load the image
    try:
        # Attempt to open the image from a filepath
        image = Image.open(image_path)
    except OSError:
        # If the image is not a local file, try to fetch it from a URL
        response = requests.get(image_path, stream=True)
        try:
            image = Image.open(response.raw)
        except IOError:
            raise ValueError(f'Failed to open image from the provided path or URL: {image_path}')

    # Create inputs for the model
    inputs = processor(images=image, text='Generate underlying data table of the figure below:', return_tensors='pt')

    # Generate the predictions
    try:
        predictions = model.generate(**inputs, max_new_tokens=512)
    except Exception as e:
        raise RuntimeError(f'Model generation failed: {str(e)}')

    # Decode the predictions to get the data table
    data_table = processor.decode(predictions[0], skip_special_tokens=True)

    # Return the extracted data table
    return data_table

# test_function_code --------------------

def test_extract_data_table_from_image():
    print("Testing started.")
    # We use a known image URL for testing
    test_image_url = 'https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png'

    # Testing case 1: Valid image URL
    print("Testing case [1/1] started.")
    try:
        result = extract_data_table_from_image(test_image_url)
        assert isinstance(result, str), f"Test case [1/1] failed: Expected a string result, got {type(result)}"
    except Exception as e:
        assert False, f"Exception occurred: {str(e)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_data_table_from_image()