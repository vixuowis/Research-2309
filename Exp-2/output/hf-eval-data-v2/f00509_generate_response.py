# function_import --------------------

from transformers import AutoModelForCausalLM

# function_code --------------------

def generate_response(input_query):
    """
    This function generates a response to a user's question using a pre-trained conversational model.

    Args:
        input_query (str): The user's question.

    Returns:
        str: The model's response.
    """
    conversation_bot = AutoModelForCausalLM.from_pretrained('Zixtrauce/JohnBot')
    output_query = conversation_bot.generate_response(input_query)
    return output_query

# test_function_code --------------------

def test_generate_response():
    """
    This function tests the 'generate_response' function by providing a sample question and checking the type of the response.
    """
    sample_query = 'What is the price of your product?'
    response = generate_response(sample_query)
    assert isinstance(response, str), 'The response should be a string.'

# call_test_function_code --------------------

test_generate_response()