# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_french_to_english(french_text):
    """
    Translates French text to English using the Helsinki-NLP/opus-mt-fr-en model.

    Args:
        french_text (str): The French text to be translated.

    Returns:
        str: The translated English text.
    """
    translator = pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
    translated_text = translator(french_text)
    english_text = translated_text[0]['translation_text']
    return english_text

# test_function_code --------------------

def test_translate_french_to_english():
    """
    Tests the translate_french_to_english function by translating a French text and checking if the output is a string.
    """
    french_text = 'Bonjour, comment ça va?'
    english_text = translate_french_to_english(french_text)
    assert isinstance(english_text, str), 'The output should be a string.'

# call_test_function_code --------------------

test_translate_french_to_english()