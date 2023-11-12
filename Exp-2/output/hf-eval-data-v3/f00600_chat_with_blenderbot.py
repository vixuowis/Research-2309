# function_import --------------------

from transformers import pipeline

# function_code --------------------

def chat_with_blenderbot(text):
    """
    This function uses the Hugging Face Transformers library to create a chatbot.
    The chatbot is capable of engaging in open-domain conversations.

    Args:
        text (str): The text message to send to the chatbot.

    Returns:
        str: The chatbot's response.

    Raises:
        OSError: If there is a problem with the disk quota.
    """
    chatbot = pipeline('conversational', model='hyunwoongko/blenderbot-9B')
    response = chatbot(text)
    return response

# test_function_code --------------------

def test_chat_with_blenderbot():
    """
    This function tests the chat_with_blenderbot function.
    It sends a variety of text messages to the chatbot and checks the type of the response.
    """
    assert isinstance(chat_with_blenderbot('What is your favorite type of music?'), str)
    assert isinstance(chat_with_blenderbot('Tell me a joke.'), str)
    assert isinstance(chat_with_blenderbot('What is the meaning of life?'), str)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_chat_with_blenderbot()