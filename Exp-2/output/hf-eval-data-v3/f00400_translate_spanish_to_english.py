# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_spanish_to_english(text):
    """
    Translates a given text from Spanish to English using the Helsinki-NLP/opus-mt-es-en model.

    Args:
        text (str): The text in Spanish that needs to be translated to English.

    Returns:
        str: The translated text in English.
    """
    translation = pipeline('translation_es_to_en', model='Helsinki-NLP/opus-mt-es-en')
    translated_text = translation(text)[0]['translation_text']
    return translated_text

# test_function_code --------------------

def test_translate_spanish_to_english():
    assert translate_spanish_to_english('Hola, ¿cómo estás?') == 'Hello, how are you?'
    assert translate_spanish_to_english('Buenos días') == 'Good morning'
    assert translate_spanish_to_english('Buenas noches') == 'Good night'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_spanish_to_english()