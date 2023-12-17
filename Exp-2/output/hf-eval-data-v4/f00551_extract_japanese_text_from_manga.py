# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_japanese_text_from_manga(image_path):
    """
    Extract Japanese text from a manga page image.

    Parameters:
        image_path (str): The file path to the manga page image.

    Returns:
        str: The extracted Japanese text.
    """
    # Create an OCR pipeline with the specified manga OCR model
    ocr_pipeline = pipeline('ocr', model='kha-white/manga-ocr-base')

    # Use the OCR pipeline to extract text from the manga page
    extracted_text = ocr_pipeline(image_path)

    return extracted_text

# test_function_code --------------------

def test_extract_japanese_text_from_manga():
    print("Testing extract_japanese_text_from_manga function.")
    sample_image_path = 'path/to/sample_manga_page.jpg'  # Replace with a valid image path

    # Expected output format (for illustration purposes, might be different in real application)
    expected_output = "ここに日本語のテキストがあります"

    # Test case 1: Check if the function returns a non-empty string
    print("Testing case [1/1] started.")
    result = extract_japanese_text_from_manga(sample_image_path)
    assert type(result) == str and result, "Test case [1/1] failed: Function should return a non-empty string."
    print("Test case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_extract_japanese_text_from_manga()