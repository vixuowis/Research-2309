def test_translate_english_to_spanish():
    """
    This function tests the 'translate_english_to_spanish' function.
    It uses a sample English text and checks if the output is not None.
    """
    # Sample English text
    text = 'Hello, how are you?'
    
    # Translate the text
    translated_text = translate_english_to_spanish(text)
    
    # Check if the output is not None
    assert translated_text is not None, 'The translation function returned None'
    
    # Print the translated text
    print(f'Translated text: {translated_text}')

test_translate_english_to_spanish()