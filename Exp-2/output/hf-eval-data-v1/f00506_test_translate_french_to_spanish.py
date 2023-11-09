def test_translate_french_to_spanish():
    """
    This function tests the translate_french_to_spanish function by comparing the output with the expected result.
    """
    # Define the test case
    test_case = 'Bonjour, comment ça va?'
    expected_result = 'Hola, ¿cómo estás?'
    
    # Call the function with the test case
    result = translate_french_to_spanish(test_case)
    
    # Assert that the result is as expected
    assert result == expected_result, f'Expected {expected_result}, but got {result}'
    
    print('All tests passed.')

test_translate_french_to_spanish()