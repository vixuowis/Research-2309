def test_get_capital_of_germany():
    '''
    This function tests the get_capital_of_germany function.
    It uses a known fact (the capital of Germany is Berlin) as a test case.
    '''
    # Call the function to get the result
    result = get_capital_of_germany()
    
    # Assert that the result is as expected
    assert result == 'Berlin', f'Error: Expected Berlin, but got {result}'

test_get_capital_of_germany()