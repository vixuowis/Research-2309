# requirements_file --------------------

!pip install -U transformers requests Pillow

# function_import --------------------

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import requests
from PIL import Image

# function_code --------------------

def analyze_chart_image(image_url):
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    processor = Pix2StructProcessor.from_pretrained('google/deplot')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(images=image, text='Generate underlying data table of the figure below:', return_tensors='pt')
    predictions = model.generate(**inputs, max_new_tokens=512)
    summary = processor.decode(predictions[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_analyze_chart_image():
    print('Testing analyze_chart_image function.')
    test_image_url = 'https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png'
    result = analyze_chart_image(test_image_url)
    assert isinstance(result, str), f'Test failed: The result should be a string, but got {type(result)}.'
    print('Test passed.')

test_analyze_chart_image()