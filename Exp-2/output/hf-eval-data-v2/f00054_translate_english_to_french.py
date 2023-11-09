# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_french(input_text):
    """
    This function translates English text to French using the Helsinki-NLP/opus-mt-en-fr model.

    Args:
        input_text (str): The text in English to be translated to French.

    Returns:
        str: The translated text in French.
    """
    translate = pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr')
    translated_text = translate(input_text)
    response = translated_text[0]['translation_text']
    return response

# test_function_code --------------------

def test_translate_english_to_french():
    """
    This function tests the translate_english_to_french function by comparing the output with the expected result.
    """
    input_text = 'Hello, how are you?'
    expected_output = 'Bonjour, comment ça va?'
    assert translate_english_to_french(input_text) == expected_output

# call_test_function_code --------------------

test_translate_english_to_french()