# function_import --------------------

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import requests
from PIL import Image

# function_code --------------------

def analyze_chart_image(image_url):
    """
    Analyze a chart image and generate a summary of the information contained within the chart.

    Args:
        image_url (str): The URL of the chart image to be analyzed.

    Returns:
        str: A summary of the information contained within the chart.
    """
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    processor = Pix2StructProcessor.from_pretrained('google/deplot')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(images=image, text='Generate underlying data table of the figure below:', return_tensors='pt')
    predictions = model.generate(**inputs, max_new_tokens=512)
    summary = processor.decode(predictions[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_analyze_chart_image():
    """
    Test the analyze_chart_image function.
    """
    image_url = 'https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png'
    summary = analyze_chart_image(image_url)
    assert isinstance(summary, str), 'The result should be a string.'
    assert len(summary) > 0, 'The result should not be an empty string.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_analyze_chart_image()