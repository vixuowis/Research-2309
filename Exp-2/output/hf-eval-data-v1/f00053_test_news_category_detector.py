def test_news_category_detector():
    """
    This function tests the news_category_detector function.
    It uses assert to validate the results.
    """
    # Define the test cases
    test_cases = [
        ("Apple just announced the newest iPhone X", "technology"),
        ("The Lakers won their last game", "sports"),
        ("The president just signed a new bill", "politics")
    ]
    
    # Test each case
    for i, (input, expected) in enumerate(test_cases):
        result = news_category_detector(input)
        assert result == expected, f'Test case {i+1} failed'
    
    print('All test cases passed')

test_news_category_detector()