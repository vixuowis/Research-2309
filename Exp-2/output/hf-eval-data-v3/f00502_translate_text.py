# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_text(text):
    """
    Translates a given text using the 'facebook/nllb-200-distilled-600M' model.

    Args:
        text (str): The text to be translated.

    Returns:
        str: The translated text.

    Raises:
        ValueError: If the model could not be loaded.
    """
    translator = pipeline('translation_xx_to_yy', model='facebook/nllb-200-distilled-600M')
    translated_text = translator(text)
    return translated_text

# test_function_code --------------------

def test_translate_text():
    """
    Tests the 'translate_text' function with some test cases.
    """
    assert translate_text('Hello World') != 'Hello World', 'Test Case 1 Failed'
    assert translate_text('Goodbye') != 'Goodbye', 'Test Case 2 Failed'
    assert translate_text('Thank you') != 'Thank you', 'Test Case 3 Failed'
    print('All Tests Passed')

# call_test_function_code --------------------

test_translate_text()