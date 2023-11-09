def test_translate_spanish_to_english():
    # Test the function with some Spanish text
    text = 'Hola, ¿cómo estás?'
    translated_text = translate_spanish_to_english(text)
    # Assert that the translated text is not empty
    assert translated_text != ''
    # Assert that the translated text is not the same as the input text
    assert translated_text != text

test_translate_spanish_to_english()