# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_german(text):
    """
    This function translates English text to German using the Hugging Face Transformers library.
    
    Args:
        text (str): The English text to be translated.
    
    Returns:
        str: The translated German text.
    """
    translator = pipeline('translation_en_to_de', model='sshleifer/tiny-marian-en-de')
    translated_text = translator(text)
    return translated_text[0]['translation_text']

# test_function_code --------------------

def test_translate_english_to_german():
    """
    This function tests the translate_english_to_german function by translating a sample English text and checking if the output is a string.
    """
    sample_text = 'Hello, how are you?'
    translated_text = translate_english_to_german(sample_text)
    assert isinstance(translated_text, str), 'The output should be a string.'

# call_test_function_code --------------------

test_translate_english_to_german()