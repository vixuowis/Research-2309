def test_translate_text():
    """
    Tests the translate_text function by translating a sample English text and checking if the output is a non-empty string.
    """
    sample_text = 'Hello, world!'
    translated_text = translate_text(sample_text)
    assert isinstance(translated_text, str)
    assert len(translated_text) > 0

test_translate_text()