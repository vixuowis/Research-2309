def test_generate_queries():
    """
    This function tests the 'generate_queries' function.
    It uses a sample document and checks if the output is a string.
    """
    # Define a sample document
    document = 'This is a sample document.'
    
    # Generate queries for the sample document
    generated_queries = generate_queries(document)
    
    # Check if the output is a string
    assert isinstance(generated_queries, str), 'The output should be a string.'
    
    # Check if the output is not empty
    assert generated_queries != '', 'The output should not be empty.'

test_generate_queries()