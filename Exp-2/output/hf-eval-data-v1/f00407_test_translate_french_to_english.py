def test_translate_french_to_english():
    '''
    Tests the translate_french_to_english function.
    '''
    # Define a French sentence and its expected English translation
    french_sentence = 'Je t’aime.'
    expected_translation = 'I love you.'

    # Call the function with the French sentence
    translation = translate_french_to_english(french_sentence)

    # Assert that the translation is close to the expected translation
    assert translation.lower() in expected_translation.lower(), f'Expected: {expected_translation}, but got: {translation}'

# Run the test function
test_translate_french_to_english()