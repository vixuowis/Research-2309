# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_italian(text):
    """
    Translate English text to Italian using the Helsinki-NLP/opus-mt-en-it model from Hugging Face Transformers.

    Args:
        text (str): The English text to be translated.

    Returns:
        str: The translated Italian text.
    """
    translator = pipeline('translation_en_to_italian', model='Helsinki-NLP/opus-mt-en-it')
    italian_text = translator(text)
    return italian_text[0]['translation_text']

# test_function_code --------------------

def test_translate_english_to_italian():
    """
    Test the translate_english_to_italian function with some English text.
    """
    english_text = 'Welcome to our website. Discover our products and services.'
    italian_text = translate_english_to_italian(english_text)
    assert isinstance(italian_text, str)

# call_test_function_code --------------------

test_translate_english_to_italian()