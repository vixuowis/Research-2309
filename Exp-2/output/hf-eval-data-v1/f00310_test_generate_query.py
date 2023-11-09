def test_generate_query():
    """
    This function tests the generate_query function.
    It uses a sample document and checks if the output is a string.
    """
    # Sample document
    document = "This is a sample document. It contains information about various topics."
    # Generate the query
    query = generate_query(document)
    # Check if the output is a string
    assert isinstance(query, str), "The output should be a string."

test_generate_query()