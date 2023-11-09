# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_table_structure(table_image):
    """
    Detects the structure (like rows, columns) in a given table image using the Hugging Face Transformers library.

    Args:
        table_image (PIL.Image.Image): The input table image.

    Returns:
        dict: The detected table structure.

    Raises:
        ValueError: If the input is not a PIL.Image.Image instance.
    """
    if not isinstance(table_image, PIL.Image.Image):
        raise ValueError('Input must be a PIL.Image.Image instance.')

    table_detector = pipeline('object-detection', model='microsoft/table-transformer-structure-recognition')
    table_structure = table_detector(table_image)
    return table_structure

# test_function_code --------------------

def test_detect_table_structure():
    """
    Tests the detect_table_structure function.
    """
    # Load a sample table image
    table_image = PIL.Image.open('sample_table_image.jpg')

    # Call the function with the sample image
    table_structure = detect_table_structure(table_image)

    # Check the type of the returned value
    assert isinstance(table_structure, dict), 'The returned value must be a dictionary.'

    # Check the keys in the returned dictionary
    expected_keys = ['boxes', 'labels', 'scores']
    assert all(key in table_structure for key in expected_keys), 'The returned dictionary must contain the keys: boxes, labels, scores.'

# call_test_function_code --------------------

test_detect_table_structure()