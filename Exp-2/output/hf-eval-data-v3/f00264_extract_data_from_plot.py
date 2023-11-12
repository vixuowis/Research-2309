# function_import --------------------

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from PIL import Image

# function_code --------------------

def extract_data_from_plot(image_path: str) -> str:
    """
    Extracts data table from a plot or chart image using the Pix2StructForConditionalGeneration model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The extracted data table in a linearized format.

    Raises:
        FileNotFoundError: If the image file does not exist.
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
    Tests the extract_data_from_plot function.
    """
    # Test with a valid image path
    image_path = 'valid_image_path.png'
    data_table = extract_data_from_plot(image_path)
    assert isinstance(data_table, str), 'The returned data table should be a string.'

    # Test with an invalid image path
    try:
        extract_data_from_plot('invalid_image_path.png')
    except FileNotFoundError:
        pass
    else:
        assert False, 'Expected a FileNotFoundError for an invalid image path.'

    print('All Tests Passed')

# call_test_function_code --------------------

test_extract_data_from_plot()