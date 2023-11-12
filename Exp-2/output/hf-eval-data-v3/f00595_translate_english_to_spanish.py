# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_spanish(text):
    """
    Translate English text to Spanish using Hugging Face Transformers.

    Args:
        text (str): The English text to be translated.

    Returns:
        str: The translated Spanish text.
    """
    translation_pipeline = pipeline('translation_en_to_es', model='Helsinki-NLP/opus-mt-en-es')
    translated_text = translation_pipeline(text)[0]['translation_text']
    return translated_text

# test_function_code --------------------

def test_translate_english_to_spanish():
    """
    Test the function translate_english_to_spanish.
    """
    assert translate_english_to_spanish('Hello, how are you?') == 'Hola, ¿cómo estás?'
    assert translate_english_to_spanish('Good morning') == 'Buenos días'
    assert translate_english_to_spanish('Good night') == 'Buenas noches'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_english_to_spanish()