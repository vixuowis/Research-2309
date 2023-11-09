def test_text_to_speech():
    '''
    This function tests the 'text_to_speech' function by using a sample text.
    '''
    # Define a sample text
    sample_text = 'This is an example sentence.'
    # Call the 'text_to_speech' function with the sample text
    audio = text_to_speech(sample_text)
    # Assert that the function returns an audio output
    assert isinstance(audio, type(None)) == False, 'The function does not return an audio output.'