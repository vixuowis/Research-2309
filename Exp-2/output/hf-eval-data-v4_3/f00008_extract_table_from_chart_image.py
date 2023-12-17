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

def extract_table_from_chart_image(chart_url):
    """
    Extracts a linearized table from a chart image.

    Args:
        chart_url (str): The URL of the chart image.

    Returns:
        str: The extracted linearized table.

    """
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    processor = Pix2StructProcessor.from_pretrained('google/deplot')

    image = Image.open(requests.get(chart_url, stream=True).raw)
    inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors='pt')
    predictions = model.generate(**inputs, max_new_tokens=512)

    table = processor.decode(predictions[0], skip_special_tokens=True)
    return table

# test_function_code --------------------

def test_extract_table_from_chart_image():
    print("Testing started.")
    # The actual URL should be replaced with a valid chart image URL
    chart_url = "https://example.com/chart_image.png"

    # Testing case 1: Extracting from a valid chart image
    print("Testing case [1/1] started.")
    table_result = extract_table_from_chart_image(chart_url)
    # The expected result should be determined by the actual outcome of the function for a known chart
    expected_result = "Expected linearized table"
    assert table_result == expected_result, f"Test case [1/1] failed: Expected {expected_result}, got {table_result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_table_from_chart_image()