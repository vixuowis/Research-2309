# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_french_to_english(text):
    """
    Translate French text to English using the Helsinki-NLP/opus-mt-fr-en model.

    Args:
        text (str): The French text to be translated.

    Returns:
        str: The translated English text.
    """
    translation_pipeline = pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
    translated_text = translation_pipeline(text)
    return translated_text[0]['translation_text']

# test_function_code --------------------

def test_translate_french_to_english():
    """
    Test the translate_french_to_english function.
    """
    assert translate_french_to_english('Bonjour') == 'Hello'
    assert translate_french_to_english('Comment ça va?') == 'How are you?'
    assert translate_french_to_english('Je suis content') == 'I am happy'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_french_to_english()