# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_response(message):
    """
    This function generates a response from a conversational model.

    Args:
        message (str): The message to which the model should respond.

    Returns:
        str: The generated response.
    """
    chatbot = pipeline('conversational', model='mywateriswet/ShuanBot')
    response = chatbot(message)
    return response

# test_function_code --------------------

def test_generate_response():
    """
    This function tests the 'generate_response' function.
    It uses a predefined message and checks if the response is not None.
    """
    message = 'What is your name?'
    response = generate_response(message)
    assert response is not None, 'The response was None.'

# call_test_function_code --------------------

test_generate_response()