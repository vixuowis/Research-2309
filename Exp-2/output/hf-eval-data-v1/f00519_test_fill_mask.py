def test_fill_mask():
    '''
    This function tests the fill_mask function.
    It uses a test dataset and asserts that the function returns the expected results.
    '''
    # Test dataset
    test_data = ['The capital of France is [MASK].', 'The largest planet in the solar system is [MASK].', 'The author of Harry Potter is [MASK].']
    
    # Expected results
    expected_results = ['The capital of France is Paris.', 'The largest planet in the solar system is Jupiter.', 'The author of Harry Potter is J.K. Rowling.']
    
    # Test the function with the test dataset
    for i, text in enumerate(test_data):
        result = fill_mask(text)
        
        # Assert that the function returns the expected results
        assert result in expected_results, f'For text: {text}, expected: {expected_results[i]}, but got: {result}'

test_fill_mask()