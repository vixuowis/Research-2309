# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_text(input_text):
    """
    Generate human-like responses to customers' questions using a pre-trained model.

    Args:
        input_text (str): The customer's question.

    Returns:
        str: The generated response.
    """
    generator = pipeline('text-generation', model='facebook/opt-350m')
    response = generator(input_text)
    return response[0]['generated_text']

# test_function_code --------------------

def test_generate_text():
    """
    Test the generate_text function.
    """
    input_text = 'What is your return policy?'
    response = generate_text(input_text)
    assert isinstance(response, str), 'The response should be a string.'
    assert len(response) > 0, 'The response should not be empty.'

# call_test_function_code --------------------

test_generate_text()