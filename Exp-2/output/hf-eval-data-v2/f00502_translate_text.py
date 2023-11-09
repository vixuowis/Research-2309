# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_text(text):
    """
    Translates the input text into a specified language using the 'facebook/nllb-200-distilled-600M' model.

    Args:
        text (str): The text to be translated.

    Returns:
        str: The translated text.
    """
    translator = pipeline('translation_xx_to_yy', model='facebook/nllb-200-distilled-600M')
    translated_text = translator(text)
    return translated_text[0]['translation_text']

# test_function_code --------------------

def test_translate_text():
    """
    Tests the 'translate_text' function by translating a sample text and checking if the output is not None.
    """
    sample_text = 'This is the content of the website.'
    translated_text = translate_text(sample_text)
    assert translated_text is not None, 'The translation function did not return any output.'

# call_test_function_code --------------------

test_translate_text()