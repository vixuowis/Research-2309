# requirements_file --------------------

!pip install -U transformers, torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_french_to_english(text):
    """
    Translate French text to English using a pre-trained translation model.

    Parameters:
    text (str): The French text to be translated.

    Returns:
    str: The translated English text.
    """
    # Load the translation model
    translator = pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
    # Perform the translation
    translation = translator(text)
    # Extract the translated text
    translated_text = translation[0]['translation_text']
    return translated_text

# test_function_code --------------------

def test_translate_french_to_english():
    print("Testing translation from French to English started.")

    # Test case 1: Hello, how are you?
    french_text = "Bonjour, comment ça va?"
    expected_english = "Hello, how are you?"
    translated_text = translate_french_to_english(french_text)
    assert translated_text == expected_english, f"Test case failed: expected '{expected_english}', got '{translated_text}'"

    print("Testing translation from French to English finished.")