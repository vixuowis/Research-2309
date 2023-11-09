def test_generate_response():
    """
    Test the generate_response function.
    """
    test_message = 'What is your name?'
    expected_response = 'I am a chatbot created by Hugging Face.'
    assert generate_response(test_message) == expected_response, 'Test failed!'

test_generate_response()