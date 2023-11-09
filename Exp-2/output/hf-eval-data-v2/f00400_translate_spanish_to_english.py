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
    """
    Tests the translate_spanish_to_english function by translating a sample text from Spanish to English.
    """
    sample_text = 'Hola, ¿cómo estás?'
    translated_text = translate_spanish_to_english(sample_text)
    assert isinstance(translated_text, str), 'The translated text should be a string.'
    assert len(translated_text) > 0, 'The translated text should not be empty.'

# call_test_function_code --------------------

test_translate_spanish_to_english()