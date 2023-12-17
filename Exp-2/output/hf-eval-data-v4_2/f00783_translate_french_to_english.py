# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_french_to_english(text: str) -> str:
    """
    Translate a given text from French to English using a pre-trained model.

    Args:
        text (str): The French text to be translated.

    Returns:
        str: The translated text in English.

    Raises:
        ValueError: If the input text is not of type string or is empty.
    """
    if not isinstance(text, str) or not text:
        raise ValueError('Input text must be a non-empty string.')
    translator = pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
    result = translator(text)
    return result[0]['translation_text']

# test_function_code --------------------

def test_translate_french_to_english():
    print('Testing started.')

    # Test case 1: Valid French string
    print('Testing case [1/2] started.')
    french_string = 'Bonjour, comment ça va?'
    translated = translate_french_to_english(french_string)
    assert isinstance(translated, str), 'Test case [1/2] failed: The result should be a string.'

    # Test case 2: Invalid input (empty string)
    print('Testing case [2/2] started.')
    try:
        translate_french_to_english('')
        assert False, 'Test case [2/2] failed: ValueError expected.'
    except ValueError as e:
        assert str(e) == 'Input text must be a non-empty string.', 'Test case [2/2] failed: Incorrect error message.'

    print('Testing finished.')

# call_test_function_line --------------------

test_translate_french_to_english()