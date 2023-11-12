# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_french_to_english(text):
    """
    Translate a French text into English using the Helsinki-NLP/opus-mt-fr-en model.

    Args:
        text (str): The French text to be translated.

    Returns:
        str: The translated English text.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string')
    translator = pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
    translated_text = translator(text)[0]['translation_text']
    return translated_text

# test_function_code --------------------

def test_translate_french_to_english():
    assert isinstance(translate_french_to_english('Bonjour'), str)
    assert 'Hello' in translate_french_to_english('Bonjour')
    assert 'Welcome' in translate_french_to_english('Bienvenue')
    try:
        translate_french_to_english(123)
    except ValueError:
        pass
    else:
        raise AssertionError('ValueError exception not raised for non-string input')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_french_to_english()