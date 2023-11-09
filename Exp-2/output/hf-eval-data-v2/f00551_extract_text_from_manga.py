# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_text_from_manga(manga_image_path):
    """
    Extracts Japanese text from a manga page using a pre-trained OCR model.

    Args:
        manga_image_path (str): The path to the image file of the manga page.

    Returns:
        str: The extracted Japanese text.

    Raises:
        FileNotFoundError: If the specified image file does not exist.
    """
    ocr_pipeline = pipeline('ocr', model='kha-white/manga-ocr-base')
    try:
        with open(manga_image_path, 'r') as file:
            extracted_text = ocr_pipeline(manga_image_path)
        return extracted_text
    except FileNotFoundError:
        print(f'File {manga_image_path} not found.')

# test_function_code --------------------

def test_extract_text_from_manga():
    """
    Tests the function extract_text_from_manga.

    Raises:
        AssertionError: If the function does not return a string.
    """
    test_image_path = 'path/to/test/image.jpg'  # Replace with the path to a test image
    result = extract_text_from_manga(test_image_path)
    assert isinstance(result, str), 'The function should return a string.'

# call_test_function_code --------------------

test_extract_text_from_manga()