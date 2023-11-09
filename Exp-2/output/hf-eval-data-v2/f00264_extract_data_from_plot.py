# function_import --------------------

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from PIL import Image

# function_code --------------------

def extract_data_from_plot(image_path):
    """
    This function extracts data tables from plots and charts using a pre-trained model from Hugging Face Transformers.

    Args:
        image_path (str): The path to the image file of the plot or chart.

    Returns:
        str: The extracted data table in a linearized format.
    """
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    processor = Pix2StructProcessor.from_pretrained('google/deplot')

    image = Image.open(image_path)

    inputs = processor(images=image, text='Generate underlying data table of the figure below:', return_tensors='pt')
    predictions = model.generate(**inputs, max_new_tokens=512)
    data_table = processor.decode(predictions[0], skip_special_tokens=True)

    return data_table

# test_function_code --------------------

def test_extract_data_from_plot():
    """
    This function tests the extract_data_from_plot function by comparing the output with the expected result.
    """
    image_path = 'test_plot_image_path.png'
    # replace 'test_plot_image_path.png' with path to your test plot or chart image

    data_table = extract_data_from_plot(image_path)

    # The expected result is dependent on the specific plot or chart image used for testing
    # Here we just check if the function returns a non-empty string
    assert isinstance(data_table, str)
    assert len(data_table) > 0

# call_test_function_code --------------------

test_extract_data_from_plot()