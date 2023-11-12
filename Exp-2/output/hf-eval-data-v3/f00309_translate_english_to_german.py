# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_german(text):
    """
    Translate English text to German using Hugging Face Transformers.

    Args:
        text (str): The English text to be translated.

    Returns:
        str: The translated German text.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')
    translator = pipeline('translation_en_to_de', model='sshleifer/tiny-marian-en-de')
    translated_text = translator(text)
    return translated_text[0]['translation_text']

# test_function_code --------------------

def test_translate_english_to_german():
    """
    Test the translate_english_to_german function.
    """
    assert translate_english_to_german('Hello world') == 'Hallo Welt'
    assert translate_english_to_german('Good morning') == 'Guten Morgen'
    assert translate_english_to_german('Good night') == 'Gute Nacht'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_english_to_german()