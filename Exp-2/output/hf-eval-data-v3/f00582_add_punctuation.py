# function_import --------------------

from transformers import pipeline

# function_code --------------------

def add_punctuation(user_message):
    """
    This function adds punctuation to a user's message using a token classification model.

    Args:
        user_message (str): The user's message that needs punctuation.

    Returns:
        str: The user's message with added punctuation.

    Raises:
        OSError: If there is an issue with loading the model or the disk quota is exceeded.
    """
    try:
        punctuator = pipeline('token-classification', model='kredor/punctuate-all')
        corrected_user_message = punctuator(user_message)
        return corrected_user_message
    except OSError as e:
        print(f'Error: {e}')

# test_function_code --------------------

def test_add_punctuation():
    """
    This function tests the add_punctuation function with a few test cases.
    """
    assert add_punctuation('hello how are you') == 'Hello, how are you?'
    assert add_punctuation('i am fine thank you') == 'I am fine, thank you.'
    assert add_punctuation('what is the weather today') == 'What is the weather today?'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_add_punctuation()