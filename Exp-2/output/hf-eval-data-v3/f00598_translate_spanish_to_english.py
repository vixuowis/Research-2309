# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_spanish_to_english(text):
    """
    Translate a Spanish text to English using the Helsinki-NLP/opus-mt-es-en model.

    Args:
        text (str): The Spanish text to be translated.

    Returns:
        str: The translated English text.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')
    translation = pipeline('translation_es_to_en', model='Helsinki-NLP/opus-mt-es-en')
    result = translation(text)
    translated_text = result[0]['translation_text']
    return translated_text

# test_function_code --------------------

def test_translate_spanish_to_english():
    assert isinstance(translate_spanish_to_english('Hola, ¿cómo estás?'), str)
    assert 'Hello, how are you?' in translate_spanish_to_english('Hola, ¿cómo estás?')
    assert 'I am fine, thank you.' in translate_spanish_to_english('Estoy bien, gracias.')
    try:
        translate_spanish_to_english(123)
    except ValueError:
        pass
    else:
        raise AssertionError('ValueError exception not raised for non-string input.')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_spanish_to_english()