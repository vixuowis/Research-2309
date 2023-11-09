def test_translate_russian_to_english():
    """
    Tests the translate_russian_to_english function by translating a sample Russian text and checking if the output is a string.
    """
    russian_text = 'Привет, мир!'
    translation = translate_russian_to_english(russian_text)
    assert isinstance(translation, str), 'The translation should be a string.'

test_translate_russian_to_english()