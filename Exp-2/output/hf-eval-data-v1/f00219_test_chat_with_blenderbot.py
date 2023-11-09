def test_chat_with_blenderbot():
    """
    This function tests the chat_with_blenderbot function by providing a sample message and checking the type of the response.
    """
    # Define a sample message
    message = 'How can I cancel my subscription?'
    
    # Call the function with the sample message
    response = chat_with_blenderbot(message)
    
    # Check that the response is a string
    assert isinstance(response, str), 'The response should be a string.'
    
    # Check that the response is not empty
    assert response != '', 'The response should not be empty.'

test_chat_with_blenderbot()