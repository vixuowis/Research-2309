# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_french(input_sentence: str) -> str:
    """
    Translate an English sentence into French using the Helsinki-NLP/opus-mt-fr-en model.

    Args:
        input_sentence (str): The English sentence to be translated.

    Returns:
        str: The translated French sentence.
    """
    translation_pipeline = pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
    translated_text = translation_pipeline(input_sentence)
    return translated_text[0]['translation_text']

# test_function_code --------------------

def test_translate_english_to_french():
    """
    Test the translate_english_to_french function with some sample sentences.
    """
    assert translate_english_to_french('Hello, how are you?') == 'Bonjour, comment ça va?'
    assert translate_english_to_french('Good morning') == 'Bonjour'
    assert translate_english_to_french('Good night') == 'Bonne nuit'

# call_test_function_code --------------------

test_translate_english_to_french()