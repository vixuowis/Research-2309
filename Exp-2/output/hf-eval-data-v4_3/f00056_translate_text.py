# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_text(text, source_lang, target_lang):
    """
    Translate a given text from the source language to the target language.

    Args:
        text (str): The text to be translated.
        source_lang (str): The code of the source language (e.g., 'en').
        target_lang (str): The code of the target language (e.g., 'fr').

    Returns:
        str: The translated text.

    Raises:
        ValueError: If the source or target language code is not supported.
    """
    translation_pipeline = pipeline(f'translation_{source_lang}_to_{target_lang}', model='facebook/nllb-200-distilled-600M')
    try:
        result = translation_pipeline(text)[0]['translation_text']
        return result
    except Exception as e:
        raise ValueError(f'An error occurred during translation: {e}')

# test_function_code --------------------

def test_translate_text():
    print("Testing started.")

    # Test case 1: Translate English to French
    print("Testing case [1/3] started.")
    translated = translate_text("Hello, how are you?", 'en', 'fr')
    assert 'Bonjour' in translated, f"Test case [1/3] failed: {translated}"

    # Test case 2: Translate Spanish to English
    print("Testing case [2/3] started.")
    translated = translate_text("Hola, cómo estás?", 'es', 'en')
    assert 'Hello' in translated, f"Test case [2/3] failed: {translated}"

    # Test case 3: Attempt to translate with unsupported language codes
    print("Testing case [3/3] started.")
    try:
        translate_text("Hola, cómo estás?", 'xx', 'yy')
        assert False, "Test case [3/3] failed: Unsupported language code exception not raised"
    except ValueError:
        assert True, "Test case [3/3] passed."

    print("Testing finished.")

# call_test_function_line --------------------

test_translate_text()