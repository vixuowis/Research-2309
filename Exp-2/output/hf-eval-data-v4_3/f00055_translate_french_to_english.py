# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_french_to_english(text):
    """
    Translates a given text from French to English using Hugging Face's Transformers library.

    Args:
        text (str): The text in French to be translated.

    Returns:
        str: The translated text in English.

    Raises:
        ValueError: If the input text is not a string or if it's empty.
    """
    if not isinstance(text, str) or not text:
        raise ValueError('Input text must be a non-empty string.')
    
    translation_pipeline = pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
    translated_text = translation_pipeline(text)
    return translated_text[0]['translation_text']

# test_function_code --------------------

def test_translate_french_to_english():
    print("Testing started.")
    # Test case 1: Valid French text
    print("Testing case [1/2] started.")
    french_text = 'Bonjour, comment ça va?'
    assert translate_french_to_english(french_text) == 'Hello, how are you?', "Test case [1/2] failed: French text was not translated correctly."

    # Test case 2: Invalid input (empty string)
    print("Testing case [2/2] started.")
    try:
        translate_french_to_english('')
        assert False, "Test case [2/2] failed: Empty string did not raise ValueError."
    except ValueError:
        assert True

    print("Testing finished.")

# call_test_function_line --------------------

test_translate_french_to_english()