def test_translate_french_to_english():
    """
    This function tests the translate_french_to_english function.
    It uses a sample French text and checks if the output is not None.
    """
    # Sample French text
    text = 'Bonjour, comment ça va?'
    
    # Translate the text
    translated_text = translate_french_to_english(text)
    
    # Check if the output is not None
    assert translated_text is not None, 'The translation function returned None'
    
    # Print the translated text
    print('Translated text:', translated_text)
    
# Run the test function
test_translate_french_to_english()