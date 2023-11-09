def test_generate_text():
    """
    This function tests the generate_text function.
    It uses a sample query and asserts that the output is not None.
    """
    # Define a sample query
    query = 'Hello, my dog is cute'
    
    # Generate the text
    output = generate_text(query)
    
    # Assert that the output is not None
    assert output is not None, 'The output is None'
    
    # Print a success message
    print('The test passed successfully.')

test_generate_text()