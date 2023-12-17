# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_website_content(text, model='facebook/nllb-200-distilled-600M'):
    """
    Translate a piece of website content to a desired language.

    Args:
        text (str): The content of the website to be translated.
        model (str): The translation model to use. Default is 'facebook/nllb-200-distilled-600M'.

    Returns:
        str: The translated text.

    Raises:
        ValueError: If the text argument is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('The text argument must be a string.')

    translator = pipeline('translation_xx_to_yy', model=model)
    translated_text = translator(text)[0]['translation_text']
    return translated_text

# test_function_code --------------------

def test_translate_website_content():
    print("Testing started.")

    # Test case 1: Translate an English sentence to another language
    print("Testing case [1/2] started.")
    english_text = 'Hello, world!'
    translated = translate_website_content(english_text)
    assert isinstance(translated, str), f"Test case [1/2] failed: The function should return a string."

    # Test case 2: Ensure passing a non-string raises ValueError
    print("Testing case [2/2] started.")
    non_string_input = 1234
    try:
        translate_website_content(non_string_input)
        assert False, "Test case [2/2] failed: The function should raise ValueError for non-string input."
    except ValueError:
        pass  # Expected behavior

    print("Testing finished.")

# call_test_function_line --------------------

test_translate_website_content()