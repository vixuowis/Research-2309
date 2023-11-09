def test_generate_motivational_sports_quote():
    '''
    This function tests the 'generate_motivational_sports_quote' function.
    It asserts that the output of the function is a string, which should be the case if the function works correctly.
    '''
    # Call the function
    result = generate_motivational_sports_quote()
    # Assert that the result is a string
    assert isinstance(result, str), 'The result should be a string.'

test_generate_motivational_sports_quote()