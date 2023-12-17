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

def analyze_chart_image(image_url: str) -> str:
    """
    Analyze a chart image and generate a summary of the information
    contained within the chart.

    Args:
        image_url: A string representing the URL of the chart image.

    Returns:
        A string summary of the image (a linearized table).

    Raises:
        ValueError: If the image URL is invalid or cannot be opened.
    """
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    processor = Pix2StructProcessor.from_pretrained('google/deplot')
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError('Failed to open image from URL.') from e

    inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
    predictions = model.generate(**inputs, max_new_tokens=512)
    summary = processor.decode(predictions[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_analyze_chart_image():
    print("Testing started.")
    test_image_url = 'https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png'

    # Test case 1: Valid image URL
    print("Testing case [1/1] started.")
    try:
        summary = analyze_chart_image(test_image_url)
        assert isinstance(summary, str), f"Test case [1/1] failed: Expected summary to be a string, got {type(summary)}"
    except Exception as e:
        assert False, f"Test case [1/1] failed with exception: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_chart_image()