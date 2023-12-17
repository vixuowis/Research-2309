# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_language(text: str) -> dict:
    """
    Detects the language of the given text using a pre-trained model.

    Args:
        text (str): The text for which to detect the language.

    Returns:
        dict: A dictionary containing the detected language and the confidence score.

    Raises:
        ValueError: If the text is empty or None.
    """
    if not text:
        raise ValueError('The text to detect language must not be empty.')

    language_detection = pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection')
    result = language_detection(text)
    return {'language': result[0]['label'], 'score': result[0]['score']}

# test_function_code --------------------

def test_detect_language():
    print("Testing started.")

    # Test case 1: English text
    print("Testing case [1/3] started.")
    result = detect_language('Hello, how are you?')
    assert result['language'] == 'en', f"Test case [1/3] failed: Expected 'en', got {result['language']}"
    assert result['score'] is not None, "Test case [1/3] failed: No confidence score returned."

    # Test case 2: French text
    print("Testing case [2/3] started.")
    result = detect_language('Bonjour, comment ça va?')
    assert result['language'] == 'fr', f"Test case [2/3] failed: Expected 'fr', got {result['language']}"

    # Test case 3: Empty text
    print("Testing case [3/3] started.")
    try:
        detect_language('')
        assert False, "Test case [3/3] failed: ValueError expected for empty text but not raised."
    except ValueError as e:
        assert str(e) == 'The text to detect language must not be empty.', f"Test case [3/3] failed: {str(e)}"

    print("Testing finished.")

# call_test_function_line --------------------

test_detect_language()