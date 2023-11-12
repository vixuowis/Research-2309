# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_text_from_manga(image_path: str) -> str:
    """
    Extracts Japanese text from a manga page image using a pre-trained OCR model.

    Args:
        image_path (str): The path to the manga page image file.

    Returns:
        str: The extracted Japanese text.

    Raises:
        FileNotFoundError: If the image file does not exist.
        KeyError: If the specified OCR model is not available.
    """
    # Create an OCR pipeline using the specified model
    ocr_pipeline = pipeline('ocr', model='kha-white/manga-ocr-base')
    # Use the pipeline to process the image and extract the text
    extracted_text = ocr_pipeline(image_path)
    return extracted_text

# test_function_code --------------------

def test_extract_text_from_manga():
    """
    Tests the extract_text_from_manga function with a sample manga page image.
    """
    # Define a path to a sample manga page image
    test_image_path = 'path/to/sample_manga_page.jpg'
    # Call the function with the test image path
    result = extract_text_from_manga(test_image_path)
    # Assert that the result is a string (the extracted text should be returned as a string)
    assert isinstance(result, str), 'The function should return the extracted text as a string.'
    # If the assertion passed, return a success message
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_text_from_manga()