def test_text_to_speech():
    '''
    This function tests the text_to_speech function by providing a sample text and checking if the output .wav file is created.
    
    Args:
    None
    
    Returns:
    None
    '''
    # Sample text for testing
    text = 'Mary had a little lamb'
    # Call the function with the sample text
    text_to_speech(text)
    # Check if the output .wav file is created
    assert os.path.exists('example_TTS.wav')