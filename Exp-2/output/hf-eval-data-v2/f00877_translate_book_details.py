# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_book_details(input_text: str) -> str:
    """
    Translates the input text from English to French using the Helsinki-NLP/opus-mt-en-fr model.

    Args:
        input_text (str): The text to be translated, in English.

    Returns:
        str: The translated text, in French.
    """
    translator = pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr')
    translated_text = translator(input_text)
    return translated_text[0]['translation_text']

# test_function_code --------------------

def test_translate_book_details():
    """
    Tests the function translate_book_details.
    """
    input_text = 'Book title and details in English...'
    translated_text = translate_book_details(input_text)
    assert isinstance(translated_text, str)

# call_test_function_code --------------------

test_translate_book_details()