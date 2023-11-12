# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_italian(text):
    """
    Translate English text to Italian using Hugging Face Transformers.

    Args:
        text (str): The English text to be translated.

    Returns:
        str: The translated Italian text.

    Raises:
        ValueError: If the input is not a string.
        OSError: If there is no space left on device for model loading.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')
    try:
        translator = pipeline('translation_en_to_it', model='Helsinki-NLP/opus-mt-en-it')
        italian_text = translator(text)[0]['translation_text']
        return italian_text
    except OSError as e:
        raise OSError('No space left on device for model loading.') from e

# test_function_code --------------------

def test_translate_english_to_italian():
    """
    Test the function translate_english_to_italian.
    """
    # Test with English text
    english_text = 'Welcome to our website. Discover our products and services.'
    italian_text = translate_english_to_italian(english_text)
    assert isinstance(italian_text, str), 'The output must be a string.'

    # Test with non-string input
    try:
        translate_english_to_italian(123)
    except ValueError as e:
        assert str(e) == 'Input text must be a string.', 'The ValueError message is incorrect.'

    # Test with large text (may cause OSError)
    large_text = 'a' * int(1e6)
    try:
        translate_english_to_italian(large_text)
    except OSError as e:
        assert str(e) == 'No space left on device for model loading.', 'The OSError message is incorrect.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_english_to_italian()