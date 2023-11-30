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
    
    if not os.path.exists(image_path):
        raise FileNotFoundError('The provided file path is invalid or empty!')
        
    # Load LayoutLMv3ForQuestionAnswering model
    model = LayoutLMv3ForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-large-uncased", revision="no_ocr").to(device)
    
    # Get property information from the image using model inference
    output = model({'image': image_path, 'question': ["What is the value?"]})
    answer = output['answers'][0]['answer'].capitalize()
        
    return answer

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