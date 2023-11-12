# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_arabic(text):
    """
    Translate English text to Arabic using Hugging Face Transformers.

    Args:
        text (str): English text to be translated.

    Returns:
        str: Translated Arabic text.
    """
    translation = pipeline('translation_en_to_arabic', model='Helsinki-NLP/opus-mt-en-ar')
    translated_text = translation(text)[0]['translation_text']
    return translated_text

# test_function_code --------------------

def test_translate_english_to_arabic():
    """
    Test the function translate_english_to_arabic.
    """
    test_text = 'My friend is planning a holiday trip for our families. He found a beautiful place with a beach, swimming pool, and a wide range of outdoor activities for kids. There\'s also a famous seafood restaurant nearby! I think our families will have a great time together.'
    expected_output = 'إن صديقي يخطط لرحلة عطلة لعائلاتنا. لقد وجد مكانًا جميلًا به شاطئ وحمام سباحة ومجموعة واسعة من الأنشطة الخارجية للأطفال. هناك أيضًا مطعم للمأكولات البحرية الشهيرة بالقرب من هنا! أعتقد أن عائلاتنا ستقضي وقتًا رائعًا معاً.'
    assert translate_english_to_arabic(test_text) == expected_output
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_english_to_arabic()