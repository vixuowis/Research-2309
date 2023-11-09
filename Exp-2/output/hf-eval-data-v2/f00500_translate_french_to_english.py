# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_french_to_english(text):
    """
    Translate a French text to English using the Helsinki-NLP/opus-mt-fr-en model.

    Args:
        text (str): The French text to be translated.

    Returns:
        str: The translated English text.
    """
    translator = pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
    translated_text = translator(text)[0]['translation_text']
    return translated_text

# test_function_code --------------------

def test_translate_french_to_english():
    """
    Test the translate_french_to_english function with some sample French texts.
    """
    french_text = 'Bonjour, comment ça va?'
    english_text = translate_french_to_english(french_text)
    assert isinstance(english_text, str), 'The translated text should be a string.'
    assert english_text != '', 'The translated text should not be empty.'

# call_test_function_code --------------------

test_translate_french_to_english()