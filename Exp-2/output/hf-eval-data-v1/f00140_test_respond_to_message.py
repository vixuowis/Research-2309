def test_respond_to_message():
    '''
    This function tests the 'respond_to_message' function by passing in a sample message and checking if the output is a string.
    '''
    input_message = 'Turn on the air conditioner.'
    response = respond_to_message(input_message)
    assert isinstance(response, str), 'The output should be a string.'

test_respond_to_message()