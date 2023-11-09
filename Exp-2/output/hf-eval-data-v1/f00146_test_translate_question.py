def test_translate_question():
    """
    This function tests the translate_question function by comparing the output with the expected result.
    """
    # Define the test input and expected output
    test_input = 'translate English to German: Where are the parks in Munich?'
    expected_output = 'Wo sind die Parks in München?'
    
    # Call the function with the test input
    output = translate_question(test_input)
    
    # Assert that the output is as expected
    assert output == expected_output, f'Expected {expected_output}, but got {output}'
    
    print('All tests passed.')

test_translate_question()