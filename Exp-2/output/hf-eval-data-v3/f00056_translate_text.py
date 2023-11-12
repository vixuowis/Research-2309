# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translates a text from one language to another using the PyTorch Transformers library.

    Args:
        text (str): The text to be translated.
        source_lang (str): The source language code.
        target_lang (str): The target language code.

    Returns:
        str: The translated text.

    Raises:
        ValueError: If the text is not a string.
        ValueError: If the source_lang or target_lang is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Text must be a string.')
    if not isinstance(source_lang, str) or not isinstance(target_lang, str):
        raise ValueError('Language codes must be strings.')

    translator = pipeline(f'translation_{source_lang}_to_{target_lang}', model='facebook/nllb-200-distilled-600M')
    translated_text = translator(text)
    return translated_text[0]['translation_text']

# test_function_code --------------------

def test_translate_text():
    assert translate_text('Hello, how are you?', 'en', 'fr') == 'Bonjour, comment ça va?'
    assert translate_text('I love you.', 'en', 'es') == 'Te quiero.'
    assert translate_text('Thank you.', 'en', 'de') == 'Danke.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_text()