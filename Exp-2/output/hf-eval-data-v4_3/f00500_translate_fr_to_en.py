# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_fr_to_en(text: str) -> str:
    """
    Translate a given text from French to English using the Helsinki-NLP translation model.

    Args:
        text: A string containing the French text to be translated.

    Returns:
        A string containing the translated English text.

    Raises:
        ValueError: If the input text is not provided.
    """
    if not text:
        raise ValueError('Input text is not provided.')
    translator = pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
    return translator(text)[0]['translation_text']

# test_function_code --------------------

def test_translate_fr_to_en():
    print("Testing started.")
    # Test case 1: Regular French sentence
    print("Testing case [1/3] started.")
    translated = translate_fr_to_en("Bonjour, comment ça va?")
    assert translated, "Test case [1/3] failed: Translation is empty."

    # Test case 2: Check empty string handling
    print("Testing case [2/3] started.")
    try:
        translate_fr_to_en("")
        assert False, "Test case [2/3] failed: Empty string should raise ValueError."
    except ValueError:
        assert True

    # Test case 3: Long French text
    print("Testing case [3/3] started.")
    translated = translate_fr_to_en("Ce long texte est pour tester la capacité du modèle à gérer de plus grandes entrées.")
    assert translated, "Test case [3/3] failed: Translation is empty."
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_fr_to_en()