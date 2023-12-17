# requirements_file --------------------

import subprocess

requirements = ["transformers", "pillow", "datasets"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_text_from_manga(manga_image):
    """
    Extracts text from a manga image using OCR.

    Args:
        manga_image (str or PIL.Image): The image to be processed.

    Returns:
        str: The extracted text from the manga image.

    Raises:
        ValueError: If manga_image is not a valid image.
    """
    if not isinstance(manga_image, (str, type(Image.open))):
        raise ValueError('Image must be a string path to the image or a PIL.Image object.')

    ocr_pipeline = pipeline('ocr', model='kha-white/manga-ocr-base')
    return ocr_pipeline(manga_image)

# test_function_code --------------------

def test_extract_text_from_manga():
    print("Testing started.")
    # Assuming 'load_dataset' and 'Image' are imported, and a dataset is available
    dataset = load_dataset("manga109s")
    sample_data = Image.open(dataset[0]["image_file_path"])  # Change field name appropriately

    # Testing case 1: Valid image input
    print("Testing case [1/3] started.")
    text = extract_text_from_manga(sample_data)
    assert type(text) == str, f"Test case [1/3] failed: Expected string, got {type(text)}"

    # Testing case 2: Testing with a non-image input
    print("Testing case [2/3] started.")
    try:
        extract_text_from_manga(123)
        assert False, "Test case [2/3] failed: ValueError not raised on non-image input."
    except ValueError as e:
        assert str(e) == 'Image must be a string path to the image or a PIL.Image object.', f"Test case [2/3] failed: {e}"

    # Testing case 3: Image path as input
    print("Testing case [3/3] started.")
    text = extract_text_from_manga(dataset[0]["image_file_path"])
    assert type(text) == str, f"Test case [3/3] failed: Expected string, got {type(text)}"

    print("Testing finished.")

# call_test_function_line --------------------

test_extract_text_from_manga()