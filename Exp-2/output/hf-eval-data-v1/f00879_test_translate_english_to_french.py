def test_translate_english_to_french():
    """
    Tests the translate_english_to_french function by translating a sample English text and checking if the output is a non-empty string.
    """
    sample_text = 'Introducing the new eco-friendly water bottle made of high-quality stainless steel with double-wall insulation to keep your drinks cool for 24 hours or hot for 12 hours.'
    translated_text = translate_english_to_french(sample_text)
    assert isinstance(translated_text, str)
    assert len(translated_text) > 0

test_translate_english_to_french()