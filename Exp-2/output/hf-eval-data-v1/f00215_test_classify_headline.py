def test_classify_headline():
    """
    This function tests the classify_headline function.
    It uses a set of predefined test cases.
    """
    # Define the test cases
    test_cases = [
        ('Apple just announced the newest iPhone X', 'technology'),
        ('The Lakers won their game last night', 'sports'),
        ('The president signed a new bill into law', 'politics')
    ]
    
    # Test the function with the test cases
    for i, (input, expected_output) in enumerate(test_cases):
        assert classify_headline(input) == expected_output, f'Test case {i+1} failed'
    
    print('All test cases passed')

test_classify_headline()