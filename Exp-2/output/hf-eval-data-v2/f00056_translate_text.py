# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_text(text, source_lang, target_lang):
    """
    Translates a given text from one language to another using the NLLB-200 model.

    Args:
        text (str): The text to be translated.
        source_lang (str): The source language code.
        target_lang (str): The target language code.

    Returns:
        str: The translated text.

    Raises:
        ValueError: If the source_lang or target_lang is not supported.
    """
    supported_languages = ['en', 'fr', 'de', 'es', 'it', 'nl', 'ru', 'el', 'tr', 'ar', 'zh', 'ja', 'ko', 'vi']
    if source_lang not in supported_languages or target_lang not in supported_languages:
        raise ValueError('Unsupported language. Supported languages are: ' + ', '.join(supported_languages))
    translator = pipeline(f'translation_{source_lang}_to_{target_lang}', model='facebook/nllb-200-distilled-600M')
    translated_text = translator(text)
    return translated_text[0]['translation_text']

# test_function_code --------------------

def test_translate_text():
    """
    Tests the translate_text function by translating a text from English to French.
    """
    text = 'Hello, how are you?'
    translated_text = translate_text(text, 'en', 'fr')
    assert isinstance(translated_text, str), 'The translated text should be a string.'
    assert len(translated_text) > 0, 'The translated text should not be empty.'

# call_test_function_code --------------------

test_translate_text()