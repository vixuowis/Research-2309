# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_text_from_manga(manga_image):
    """
    Extracts Japanese text from a manga image using the Manga OCR API.

    Args:
        manga_image (str): Path to the manga image file.

    Returns:
        str: The extracted text from the manga image.

    Raises:
        Exception: If the image file does not exist or the OCR pipeline fails.
    """
    try:
        ocr_pipeline = pipeline('ocr', model='kha-white/manga-ocr-base')
        manga_text = ocr_pipeline(manga_image)
        return manga_text
    except Exception as e:
        print(f'Error: {e}')
        raise

# test_function_code --------------------

def test_extract_text_from_manga():
    """
    Tests the function extract_text_from_manga.
    """
    test_image = 'test_manga_image.jpg'
    result = extract_text_from_manga(test_image)
    assert isinstance(result, str), 'The result should be a string.'

# call_test_function_code --------------------

test_extract_text_from_manga()