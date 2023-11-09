def test_generate_response():
    # Test the generate_response function
    # Input: A test question
    # Output: The model's response
    input_query = 'What is the price of your product?'
    output_query = generate_response(input_query)
    # Assert that the output is a string (the model's response)
    assert isinstance(output_query, str)

test_generate_response()