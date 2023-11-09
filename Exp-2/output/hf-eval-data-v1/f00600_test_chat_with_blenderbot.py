# Test function for chat_with_blenderbot
# The function takes no arguments
# It uses assert to verify the function's output

def test_chat_with_blenderbot():
    # Define a test case
    test_case = 'What is your favorite type of music?'
    # Call the function with the test case
    response = chat_with_blenderbot(test_case)
    # Assert that the response is not None
    assert response is not None
    # Assert that the response is a string
    assert isinstance(response, str)
    # Print a success message
    print('Test passed!')

# Call the test function
test_chat_with_blenderbot()