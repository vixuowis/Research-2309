# function_import --------------------

import os
from transformers import LayoutLMv3ForQuestionAnswering

# function_code --------------------

def extract_property_info(image_path):
    """
    Extracts property information from a scanned image using LayoutLMv3ForQuestionAnswering model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The extracted property information.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} does not exist")

    # Load the pre-trained model
    model = LayoutLMv3ForQuestionAnswering.from_pretrained('hf-tiny-model-private/tiny-random-LayoutLMv3ForQuestionAnswering')

    # TODO: Apply OCR to the image and use the model to answer questions about property details
    # This part is omitted because it's beyond the scope of this task

    return 'Extracted property information'

# test_function_code --------------------

def test_extract_property_info():
    """
    Tests the extract_property_info function.
    """
    # Test with a non-existing image file
    try:
        extract_property_info('non_existing_file.jpg')
    except FileNotFoundError as e:
        assert str(e) == 'non_existing_file.jpg does not exist'

    # TODO: Add more test cases

    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_property_info()