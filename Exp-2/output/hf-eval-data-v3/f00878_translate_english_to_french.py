# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_french(input_sentence: str) -> str:
    '''
    Translates an English sentence into French using the Helsinki-NLP/opus-mt-fr-en model.

    Args:
        input_sentence (str): The English sentence to be translated.

    Returns:
        str: The translated French sentence.

    Raises:
        ValueError: If the input is not a string.
    '''
    if not isinstance(input_sentence, str):
        raise ValueError('Input sentence must be a string.')
    translation_pipeline = pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
    translated_text = translation_pipeline(input_sentence)
    return translated_text[0]['translation_text']

# test_function_code --------------------

def test_translate_english_to_french():
    '''
    Tests the translate_english_to_french function with some example sentences.
    '''
    assert translate_english_to_french('Hello, how are you?') == 'Bonjour, comment ça va?'
    assert translate_english_to_french('Good morning') == 'Bonjour'
    assert translate_english_to_french('Good night') == 'Bonne nuit'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_english_to_french()