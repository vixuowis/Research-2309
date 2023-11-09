def test_translate_english_to_german():
    """
    This function tests the 'translate_english_to_german' function.
    It uses a sample English text and checks if the output is not None.
    """
    # Sample English text
    text = 'Hello, how are you?'
    # Translate the text
    translated_text = translate_english_to_german(text)
    # Check if the output is not None
    assert translated_text is not None, 'The translation function returned None'
    # Print the translated text
    print('Translated text:', translated_text)

test_translate_english_to_german()