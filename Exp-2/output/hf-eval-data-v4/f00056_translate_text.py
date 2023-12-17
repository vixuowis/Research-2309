# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_text(text, source_language='en', target_language='fr'):
    """
    Translates a given text from a source language to a target language using the NLLB-200 model.

    :param text: String, the text to translate.
    :param source_language: String, the language code of the source text (default='en').
    :param target_language: String, the language code of the target language (default='fr').
    :return: String, the translated text.
    """
    # Initialize the translation pipeline
    translator = pipeline(f'translation_{source_language}_to_{target_language}', model='facebook/nllb-200-distilled-600M')
    # Translate the text
    translated_result = translator(text)
    # Extract translated text
    translated_text = translated_result[0]['translation_text']
    return translated_text

# test_function_code --------------------

def test_translate_text():
    print("Testing started.")

    # Test case 1: Translate English to French
    english_text = "Hello, how are you?"
    french_text = translate_text(english_text, 'en', 'fr')
    print("Testing case [1/1] started.")
    assert isinstance(french_text, str), f"Test case [1/1] failed: The result should be a string, but got {type(french_text)} instead."
    print(f"Translation result: {french_text}")
    print("Testing case [1/1] finished.")
    print("Testing finished.")

# Run the test function
test_translate_text()