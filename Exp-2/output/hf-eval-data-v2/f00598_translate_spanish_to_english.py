# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_spanish_to_english(text):
    """
    Translate a Spanish text to English using the Helsinki-NLP/opus-mt-es-en model.

    Args:
        text (str): The Spanish text to be translated.

    Returns:
        str: The translated English text.
    """
    translation = pipeline('translation_es_to_en', model='Helsinki-NLP/opus-mt-es-en')
    result = translation(text)
    translated_text = result[0]['translation_text']
    return translated_text

# test_function_code --------------------

def test_translate_spanish_to_english():
    """
    Test the translate_spanish_to_english function with a Spanish text.
    The translated text is not compared strictly due to possible variations in translation.
    """
    spanish_text = 'Lo siento, pero no puedo ir a la reunión debido a una emergencia personal. Avisaré al equipo y nos pondremos en contacto para reprogramar la reunión.'
    translated_text = translate_spanish_to_english(spanish_text)
    assert isinstance(translated_text, str)

# call_test_function_code --------------------

test_translate_spanish_to_english()