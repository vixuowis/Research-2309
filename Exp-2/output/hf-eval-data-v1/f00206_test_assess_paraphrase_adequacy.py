def test_assess_paraphrase_adequacy():
    """
    This function tests the 'assess_paraphrase_adequacy' function.
    """
    # Define a test paraphrase
    test_paraphrase = 'The quick brown fox jumps over the lazy dog.'
    
    # Call the 'assess_paraphrase_adequacy' function with the test paraphrase
    test_result = assess_paraphrase_adequacy(test_paraphrase)
    
    # Assert that the result is a dictionary (as expected)
    assert isinstance(test_result, dict), 'The result should be a dictionary.'
    
    # Assert that the dictionary contains a 'label' and a 'score'
    assert 'label' in test_result, 'The result should contain a label.'
    assert 'score' in test_result, 'The result should contain a score.'
    
    # Assert that the score is a float
    assert isinstance(test_result['score'], float), 'The score should be a float.'
    
    # Assert that the score is within the expected range (0.0 to 1.0)
    assert 0.0 <= test_result['score'] <= 1.0, 'The score should be between 0.0 and 1.0.'

test_assess_paraphrase_adequacy()