# function_import --------------------

from transformers import pipeline

# function_code --------------------

def chat_with_blenderbot(message):
    """
    This function uses the Hugging Face Transformers library to create a chatbot that can engage in open-domain conversations.
    The chatbot is based on the 'hyunwoongko/blenderbot-9B' model, which has been trained on a variety of dialogue datasets.

    Args:
        message (str): The message that will be sent to the chatbot.

    Returns:
        str: The chatbot's response.
    """
    chatbot = pipeline('conversational', model='hyunwoongko/blenderbot-9B')
    response = chatbot(message)
    return response

# test_function_code --------------------

def test_chat_with_blenderbot():
    """
    This function tests the 'chat_with_blenderbot' function by sending a message to the chatbot and checking if a response is received.
    """
    response = chat_with_blenderbot('What is your favorite type of music?')
    assert isinstance(response, str), 'The response from the chatbot should be a string.'

# call_test_function_code --------------------

test_chat_with_blenderbot()