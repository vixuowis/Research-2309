# requirements_file --------------------

!pip install -U transformers requests Pillow

# function_import --------------------

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import requests
from PIL import Image

# function_code --------------------

def extract_table_from_chart(image_url):
    """
    Extracts a linearized data table from a given chart image URL using a pre-trained model.

    Args:
        image_url (str): URL of the chart image.

    Returns:
        str: Linearized table as text.
    """
    # Load the pre-trained DePlot model and processor from Hugging Face Transformers
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    processor = Pix2StructProcessor.from_pretrained('google/deplot')

    # Open the image from the URL
    response = requests.get(image_url, stream=True)
    image = Image.open(response.raw)

    # Prepare the image for the model
    inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors='pt')

    # Generate predictions
    predictions = model.generate(**inputs, max_new_tokens=512)

    # Decode the predictions to obtain the linearized table
    table = processor.decode(predictions[0], skip_special_tokens=True)
    return table

# test_function_code --------------------

def test_extract_table_from_chart():
    print("Testing started.")
    test_url = 'https://example.com/test_chart.png'  # Example URL for testing

    # Expected output is unknown for a random image, so we test if the function is error-free
    try:
        table_text = extract_table_from_chart(test_url)
        assert isinstance(table_text, str), "The output must be a string."
        print("Test passed: Function returned a string.")
    except Exception as e:
        assert False, f"Test failed: {e}"

    print("Testing finished.")

test_extract_table_from_chart()