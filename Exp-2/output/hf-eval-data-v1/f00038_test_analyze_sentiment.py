def test_analyze_sentiment():
    """
    This function tests the 'analyze_sentiment' function with a sample review.
    It uses the 'assert' statement to verify that the function returns a result.
    """
    # Define a sample review
    review = '¡Esto es maravilloso! Me encanta.'
    
    # Call the 'analyze_sentiment' function with the sample review
    result = analyze_sentiment(review)
    
    # Assert that the function returns a result
    assert result is not None, 'The function did not return a result.'
    
    # Assert that the result is a list (as the pipeline function returns a list of results)
    assert isinstance(result, list), 'The function did not return a list.'
    
    # Assert that the list is not empty
    assert len(result) > 0, 'The function returned an empty list.'
    
    # Assert that the first item in the list is a dictionary (as each result is a dictionary)
    assert isinstance(result[0], dict), 'The function did not return a dictionary.'
    
    # Assert that the dictionary contains a 'label' key (as each result should have a 'label' key)
    assert 'label' in result[0], 'The function did not return a result with a label.'
    
    # Call the test function
    test_analyze_sentiment()