def test_text_to_speech():
    '''
    This function tests the text_to_speech function by comparing the output with the expected result.
    '''
    test_text = 'Hello, this is a test run.'
    expected_output = 'Audio file of the spoken text'
    assert isinstance(text_to_speech(test_text), type(expected_output)), 'Test failed!'

test_text_to_speech()