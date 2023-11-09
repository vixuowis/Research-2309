def test_detect_named_entities():
    """
    This function tests the detect_named_entities function by using a sample text.
    The function asserts that the returned result is a list.
    """
    # Define a sample text
    sample_text = 'Nader Jokhadar had given Syria the lead with a well-struck header in the seventh minute.'
    
    # Call the detect_named_entities function with the sample text
    result = detect_named_entities(sample_text)
    
    # Assert that the result is a list
    assert isinstance(result, list), 'The result should be a list.'
    
    # Assert that the list is not empty
    assert len(result) > 0, 'The list should not be empty.'
    
    # Assert that each item in the list is a dictionary
    for item in result:
        assert isinstance(item, dict), 'Each item in the list should be a dictionary.'

test_detect_named_entities()