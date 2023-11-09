def test_translate_english_to_italian():
    """
    This function tests the translate_english_to_italian function by translating a sample English text and checking if the output is a string.
    """
    # Define a sample English text
    sample_text = 'Welcome to our website. Discover our products and services.'
    
    # Translate the sample text to Italian
    translated_text = translate_english_to_italian(sample_text)
    
    # Check if the translated text is a string
    assert isinstance(translated_text, str), 'The translated text should be a string.'
    
    # Print the translated text for manual verification
    print(f'Translated text: {translated_text}')

test_translate_english_to_italian()