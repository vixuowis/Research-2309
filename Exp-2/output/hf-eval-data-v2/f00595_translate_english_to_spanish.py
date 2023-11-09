# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_spanish(text):
    """
    Translate English text to Spanish using the Hugging Face Transformers library.

    Args:
        text (str): The English text to be translated.

    Returns:
        str: The translated Spanish text.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')
    translation_pipeline = pipeline('translation_en_to_es', model='Helsinki-NLP/opus-mt-en-es')
    translated_text = translation_pipeline(text)
    return translated_text[0]['translation_text']

# test_function_code --------------------

def test_translate_english_to_spanish():
    """
    Test the function translate_english_to_spanish.
    """
    test_text = 'Hello, how are you?'
    translated_text = translate_english_to_spanish(test_text)
    assert isinstance(translated_text, str)

# call_test_function_code --------------------

test_translate_english_to_spanish()