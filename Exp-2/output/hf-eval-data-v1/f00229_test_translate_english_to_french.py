def test_translate_english_to_french():
    # Test the translate_english_to_french function with some English text
    english_text = 'This is a contract.'
    french_text = translate_english_to_french(english_text)

    # Assert that the translated text is not the same as the original English text
    assert english_text != french_text, 'The translated text is the same as the original text.'

    # Assert that the translated text is not empty
    assert len(french_text) > 0, 'The translated text is empty.'

    # Print the original and translated text for reference
    print('Original text:', english_text)
    print('Translated text:', french_text)

test_translate_english_to_french()