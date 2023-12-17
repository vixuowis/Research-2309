# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def create_chatbot_response(user_message: str) -> str:
    """
    Create a response from the chatbot based on the user's message.

    Args:
        user_message (str): The message from the user to which the chatbot should respond.

    Returns:
        str: The chatbot's generated response.

    Raises:
        ValueError: If the user_message is empty or None.
    """
    if not user_message:
        raise ValueError('The user_message must not be empty.')

    # Initialize the chatbot using the pre-trained Blenderbot model
    chatbot = pipeline('conversational', model='hyunwoongko/blenderbot-9B')

    # Generate the response
    response = chatbot(user_message)

    # Return the text of the generated response
    return response[0]['generated_text']

# test_function_code --------------------

def test_create_chatbot_response():
    print("Testing started.")

    # Test case 1: Valid input message
    print("Testing case [1/2] started.")
    response = create_chatbot_response("Hello, can you give me travel tips?")
    assert type(response) == str and len(response) > 0, f"Test case [1/2] failed: Expected a string response, got {type(response)}"

    # Test case 2: Empty input message
    print("Testing case [2/2] started.")
    try:
        create_chatbot_response("")
        assert False, "Test case [2/2] failed: Expected ValueError for empty message."
    except ValueError as e:
        assert str(e) == 'The user_message must not be empty.', f"Test case [2/2] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_create_chatbot_response()