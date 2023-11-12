# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_text(input_text: str, model: str = 'Helsinki-NLP/opus-mt-en-fr') -> str:
    """
    Translates the input text from English to French using the specified model.

    Args:
        input_text (str): The text to be translated.
        model (str, optional): The translation model to be used. Defaults to 'Helsinki-NLP/opus-mt-en-fr'.

    Returns:
        str: The translated text.
    """
    translator = pipeline('translation_en_to_fr', model=model)
    translated_text = translator(input_text)
    return translated_text[0]['translation_text']

# test_function_code --------------------

def test_translate_text():
    """
    Tests the translate_text function with some test cases.
    """
    assert translate_text('Hello, world!') == 'Bonjour, monde!'
    assert translate_text('Good morning') == 'Bon matin'
    assert translate_text('Good night') == 'Bonne nuit'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_text()