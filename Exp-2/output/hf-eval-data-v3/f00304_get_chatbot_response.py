# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_chatbot_response(user_input):
    """
    This function uses the Hugging Face Transformers library to generate a response from a conversational AI model.

    Args:
        user_input (str): The user's message to the chatbot.

    Returns:
        str: The chatbot's response.
    """
    chatbot = pipeline('conversational', model='hyunwoongko/blenderbot-9B')
    response = chatbot(user_input)
    return response['generated_text']

# test_function_code --------------------

def test_get_chatbot_response():
    """
    This function tests the get_chatbot_response function.
    """
    test_input_1 = 'I am planning a vacation to Italy. Can you suggest some must-visit places?'
    test_input_2 = 'What is the weather like in New York?'
    test_input_3 = 'Can you recommend some good books to read during travel?'

    assert isinstance(get_chatbot_response(test_input_1), str)
    assert isinstance(get_chatbot_response(test_input_2), str)
    assert isinstance(get_chatbot_response(test_input_3), str)

    print('All Tests Passed')

# call_test_function_code --------------------

test_get_chatbot_response()