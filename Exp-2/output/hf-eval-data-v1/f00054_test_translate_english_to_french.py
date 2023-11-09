def test_translate_english_to_french():
    # Test the translate_english_to_french function
    # The test dataset is the OPUS dataset
    # However, for simplicity, we will use a few English sentences
    # and their corresponding French translations
    
    test_data = [
        ('Hello, how are you?', 'Bonjour, comment ça va?'),
        ('I am fine, thank you.', 'Je vais bien, merci.'),
        ('What is your name?', 'Comment vous appelez-vous?')
    ]
    
    for english, french in test_data:
        # Translate the English text to French
        translation = translate_english_to_french(english)
        
        # Check if the translation is close to the expected French translation
        # We use the in operator instead of == to avoid strict comparison
        assert french in translation, f'Expected {french}, but got {translation}'
    
    print('All tests passed.')

# Run the test function
test_translate_english_to_french()