# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_arabic(text: str) -> str:
    """
    Translate English text to Arabic using Hugging Face Transformers.

    Args:
        text (str): The English text to be translated.

    Returns:
        str: The translated Arabic text.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')
    translation = pipeline('translation_en_to_ar', model='Helsinki-NLP/opus-mt-en-ar')
    translated_text = translation(text)
    return translated_text[0]['translation_text']

# test_function_code --------------------

def test_translate_english_to_arabic():
    """
    Test the function translate_english_to_arabic.
    """
    test_text = 'My friend is planning a holiday trip for our families.'
    expected_output = 'إن صديقي يخطط لرحلة عطلة لعائلاتنا.'
    assert translate_english_to_arabic(test_text) == expected_output
    test_text = 'He found a beautiful place with a beach, swimming pool, and a wide range of outdoor activities for kids.'
    expected_output = 'لقد وجد مكانًا جميلًا به شاطئ وحمام سباحة ومجموعة واسعة من الأنشطة الخارجية للأطفال.'
    assert translate_english_to_arabic(test_text) == expected_output

# call_test_function_code --------------------

test_translate_english_to_arabic()