def test_extract_named_entities():
    '''
    This function tests the 'extract_named_entities' function.
    '''
    # Define a test case
    text = 'On September 1st George Washington won 1 dollar.'
    
    # Call the function with the test case
    result = extract_named_entities(text)
    
    # Assert that the function returns the expected result
    assert len(result) > 0, 'No named entities were extracted.'
    assert any('George Washington' in str(entity) for entity in result), 'Expected named entity not found.'
    
    print('All test cases pass')
    
# Call the test function
test_extract_named_entities()