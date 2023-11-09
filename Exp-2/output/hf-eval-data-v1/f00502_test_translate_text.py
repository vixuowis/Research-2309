def test_translate_text():
    """
    This function tests the 'translate_text' function by translating a sample text.
    The translated text is not compared strictly due to the nature of language translation.
    """
    # Sample text to be translated
    text = 'Hello World'
    
    # Translate the text
    translated_text = translate_text(text)
    
    # Check if the translated text is not empty
    assert translated_text != '', 'The translation function did not return any text.'
    
    # Check if the translated text is not the same as the input text
    assert translated_text != text, 'The translation function did not change the input text.'

test_translate_text()