def test_generate_houseplant_care_tips():
    """
    This function tests the 'generate_houseplant_care_tips' function.
    It uses a fixed prompt and checks if the output is a non-empty string.
    """
    # Define a test prompt
    test_prompt = 'Tips on how to take care of houseplants:'
    
    # Call the function with the test prompt
    result = generate_houseplant_care_tips(test_prompt)
    
    # Check if the result is a non-empty string
    assert isinstance(result, str), 'The result should be a string.'
    assert len(result) > 0, 'The result should not be an empty string.'

test_generate_houseplant_care_tips()