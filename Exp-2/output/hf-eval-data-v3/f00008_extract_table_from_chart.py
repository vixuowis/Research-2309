# function_import --------------------

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import requests
from PIL import Image

# function_code --------------------

def extract_table_from_chart(chart_url):
    """
    Extracts a linearized table from a chart image.

    Args:
        chart_url (str): The URL of the chart image.

    Returns:
        str: The linearized table extracted from the chart.

    Raises:
        requests.exceptions.RequestException: If there is an error to make a GET request to the chart_url.
        urllib3.exceptions.ProtocolError: If there is a network problem (e.g. a timeout, dropped connection, etc).
        requests.exceptions.ChunkedEncodingError: If the server declared chunked encoding but sent an invalid chunk.
    """
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    processor = Pix2StructProcessor.from_pretrained('google/deplot')

    image = Image.open(requests.get(chart_url, stream=True).raw)

    inputs = processor(images=image, text='Generate underlying data table of the figure below:', return_tensors='pt')
    predictions = model.generate(**inputs, max_new_tokens=512)

    return processor.decode(predictions[0], skip_special_tokens=True)

# test_function_code --------------------

def test_extract_table_from_chart():
    """
    Tests the extract_table_from_chart function with a few test cases.
    """
    chart_url = 'https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png'
    result = extract_table_from_chart(chart_url)
    assert isinstance(result, str), 'The result should be a string.'

    chart_url = 'https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5091.png'
    result = extract_table_from_chart(chart_url)
    assert isinstance(result, str), 'The result should be a string.'

    chart_url = 'https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5092.png'
    result = extract_table_from_chart(chart_url)
    assert isinstance(result, str), 'The result should be a string.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_table_from_chart()