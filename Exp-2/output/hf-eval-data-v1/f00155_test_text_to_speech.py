def test_text_to_speech():
    '''
    This function tests the text_to_speech function with a sample text.
    '''
    # Define the sample text
    text = 'Bonjour, ceci est un test.'
    # Call the text_to_speech function with the sample text
    wav, rate = text_to_speech(text)
    # Assert that the output is not None
    assert wav is not None
    assert rate is not None

test_text_to_speech()