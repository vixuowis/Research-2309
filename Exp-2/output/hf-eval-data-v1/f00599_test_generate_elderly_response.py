def test_generate_elderly_response():
    '''
    This function tests the generate_elderly_response function.
    It uses a sample question and checks if the response is a string.
    '''
    # Define a sample question
    sample_question = "You: What advice would you give to someone just starting their career?"
    # Call the function with the sample question
    response = generate_elderly_response(sample_question)
    # Check if the response is a string
    assert isinstance(response, str), "The response should be a string."
    # Check if the response is not empty
    assert response != "", "The response should not be empty."

test_generate_elderly_response()