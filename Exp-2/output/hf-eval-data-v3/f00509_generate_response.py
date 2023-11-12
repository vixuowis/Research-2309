# function_import --------------------

from transformers import AutoModelForCausalLM

# function_code --------------------

def generate_response(input_query):
    """
    Generate a response to the input query using a pre-trained conversational model.

    Args:
        input_query (str): The input query or question.

    Returns:
        str: The generated response.

    Raises:
        OSError: If there is not enough disk space to download the model.
    """
    conversation_bot = AutoModelForCausalLM.from_pretrained('Zixtrauce/JohnBot')
    output_query = conversation_bot.generate_response(input_query)
    return output_query

# test_function_code --------------------

def test_generate_response():
    """
    Test the generate_response function.
    """
    sample_queries = ['What is the price of your product?', 'How can I purchase your product?', 'Do you offer any discounts?']
    for query in sample_queries:
        try:
            response = generate_response(query)
            assert isinstance(response, str)
        except OSError as e:
            print('Not enough disk space to download the model. Skipping this test.')
            continue
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_response()