# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_text_from_manga_image(manga_image_path):
    """
    Extract Japanese text from a manga image using OCR technology.

    Parameters:
    manga_image_path (str): The file path to the manga image from which to extract text.

    Returns:
    str: The extracted text as a string.
    """
    # Initialize the OCR pipeline using the manga-ocr-base model
    ocr_pipeline = pipeline('ocr', model='kha-white/manga-ocr-base')

    # Open the manga image
    with open(manga_image_path, 'rb') as image_file:
        manga_image = image_file.read()

    # Extract text from the image using the OCR pipeline
    extracted_text = ocr_pipeline(manga_image)
    return extracted_text

# test_function_code --------------------

def test_extract_text_from_manga_image():
    print("Testing started.")

    # Test case 1: Extract text from a sample manga image
    print("Testing case [1/1] started.")
    extracted_text = extract_text_from_manga_image('sample_manga_page.jpg')
    assert isinstance(extracted_text, str), f"Test case [1/1] failed: Expected string output, got {type(extracted_text)}"
    print("Testing finished.")

# Run the test function
test_extract_text_from_manga_image()