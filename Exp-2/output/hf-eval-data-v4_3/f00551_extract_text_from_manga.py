# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_text_from_manga(manga_image_path):
    """
    Extracts Japanese text from a manga page image.

    Args:
        manga_image_path (str): The file path to the manga page image.

    Returns:
        str: The extracted Japanese text from the image.

    Raises:
        FileNotFoundError: If the given manga_image_path does not exist.
        Exception: If the OCR model fails to process the image.
    """
    # Check if the file exists
    if not os.path.isfile(manga_image_path):
        raise FileNotFoundError(f'No file found at {manga_image_path}')

    # Initialize the manga OCR model
    ocr_pipeline = pipeline('ocr', model='kha-white/manga-ocr-base')

    # Try to extract text using the OCR model
    try:
        extracted_text = ocr_pipeline(manga_image_path)
    except Exception as e:
        raise Exception(f'OCR model failed: {e}')

    return extracted_text['text']

# test_function_code --------------------

def test_extract_text_from_manga():
    print("Testing started.")

    # The manga page for testing must be available at 'tests/test_manga_page.jpg'
    manga_image_path = 'tests/test_manga_page.jpg'

    # Test case 1: Valid manga page image
    print("Testing case [1/1] started.")
    try:
        extracted_text = extract_text_from_manga(manga_image_path)
        assert isinstance(extracted_text, str), f"Test case [1/1] failed: Expected string output, got {type(extracted_text)}"
    except Exception as e:
        print(f"Test case [1/1] failed: {e}")
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_text_from_manga()