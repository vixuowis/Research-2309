def test_classify_message():
    '''
    This function tests the 'classify_message' function with some example messages.
    '''
    assert classify_message('Hello, how are you?') == 'Safe message.'
    assert classify_message('You are stupid.') == 'Warning: Inappropriate message detected.'

test_classify_message()