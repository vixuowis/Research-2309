def test_text_to_speech():
    '''
    This function tests the text_to_speech function.
    '''
    # Define the test text
    test_text = 'Hello, this is a test run.'
    # Call the text_to_speech function
    audio_output = text_to_speech(test_text)
    # Assert that the output is not None
    assert audio_output is not None, 'The output is None.'
    # Assert that the output is an instance of IPython.lib.display.Audio
    assert isinstance(audio_output, ipd.lib.display.Audio), 'The output is not an instance of IPython.lib.display.Audio.'

test_text_to_speech()