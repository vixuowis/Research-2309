def test_translate_french_to_english():
    """
    This function tests the translate_french_to_english function by comparing the output with the expected result.
    """
    # Define the test text and the expected result
    test_text = 'Bonjour, comment ça va?'
    expected_result = 'Hello, how are you?'
    
    # Call the function with the test text
    result = translate_french_to_english(test_text)
    
    # Assert that the result is as expected
    assert result.lower() in expected_result.lower(), f'Expected {expected_result}, but got {result}'

test_translate_french_to_english()