def test_generate_response():
    """
    Test the generate_response function.
    """
    user_input = 'What are the benefits of regular exercise?'
    response = generate_response(user_input)
    assert isinstance(response, str), 'The response should be a string.'
    assert len(response) > 0, 'The response should not be empty.'

test_generate_response()