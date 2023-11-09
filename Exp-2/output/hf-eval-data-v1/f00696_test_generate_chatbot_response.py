def test_generate_chatbot_response():
    # Test the generate_chatbot_response function
    # The function should return a string response to the input message
    # The response should be a friendly conversation, answering the user's query
    # The test function uses assert to check if the function works as expected
    # The test function does not compare numbers strictly
    # The test function uses a sample input message for testing
    input_message = 'Hello, how are you?'
    output = generate_chatbot_response(input_message)
    assert isinstance(output, str), 'Output should be a string'
    assert output != '', 'Output should not be an empty string'

test_generate_chatbot_response()