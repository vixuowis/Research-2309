# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_customer_support_response(question_text):
    """
    Generate a human-like response to a customer's question using a pre-trained text generation model.

    Args:
        question_text (str): The customer's question to which the chatbot should respond.

    Returns:
        str: A human-like response generated by the model.

    Raises:
        ValueError: If the question_text is not a string.
    """
    if not isinstance(question_text, str):
        raise ValueError('Input question must be a string.')
    
    # Load the pre-trained text generation model
    generator = pipeline('text-generation', model='facebook/opt-350m')
    
    # Generate a response
    responses = generator(question_text, num_return_sequences=1)
    return responses[0]['generated_text']

# test_function_code --------------------

def test_generate_customer_support_response():
    print("Testing started.")
    # Test case 1: Valid question string
    print("Testing case [1/3] started.")
    response = generate_customer_support_response('What is your return policy?')
    assert isinstance(response, str), f"Test case [1/3] failed: The response should be a string."

    # Test case 2: Handle non-string input
    print("Testing case [2/3] started.")
    try:
        response = generate_customer_support_response(None)
    except ValueError as e:
        assert str(e) == 'Input question must be a string.', f"Test case [2/3] failed: Expected a ValueError for non-string input."

    # Test case 3: Inspect the response content
    print("Testing case [3/3] started.")
    response = generate_customer_support_response('How can I track my order?')
    assert len(response) > 0, f"Test case [3/3] failed: The response should not be empty."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_customer_support_response()