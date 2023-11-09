def test_generate_response():
    """
    Test the generate_response function.
    """
    # Define a test question
    test_question = 'What is your return policy?'

    # Generate a response to the test question
    response = generate_response(test_question)

    # Assert that the response is a string
    assert isinstance(response, str), 'Response must be a string.'

    # Assert that the response is not empty
    assert response != '', 'Response must not be empty.'

    # Assert that the response contains the test question
    assert test_question in response, 'Response must contain the test question.'

test_generate_response()