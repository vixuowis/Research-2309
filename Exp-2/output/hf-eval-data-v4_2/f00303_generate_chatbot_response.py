# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_chatbot_response(input_message: str) -> str:
    """
    Generate a response from the chatbot based on the input message.

    Args:
        input_message (str): The user input message to which the chatbot should respond.

    Returns:
        str: The generated response from the chatbot.

    Raises:
        ValueError: If the input message is empty.
    """
    if not input_message:
        raise ValueError('Input message cannot be empty.')

    chatbot = pipeline('text-generation', 'PygmalionAI/pygmalion-1.3b')
    character_persona = "CompanyBot's Persona: I am a helpful chatbot designed to answer questions about our products and services."
    dialogue_history = ''

    input_prompt = f'{character_persona}\n{dialogue_history}You: {input_message}\n'
    response = chatbot(input_prompt)

    # Assuming the response is a list of generated texts, return the first one
    return response[0]['generated_text']

# test_function_code --------------------

def test_generate_chatbot_response():
    print('Testing started.')
    sample_input = 'What products do you offer?'
    expected_output_contains = ['Our products', 'We offer']  # This is an assumption, the test might need to be adapted depending on the model's actual output

    # Test case 1: Non-empty input
    print('Testing case [1/2] started.')
    response = generate_chatbot_response(sample_input)
    assert any(x in response for x in expected_output_contains), f'Test case [1/2] failed: The response does not contain expected product information. Response was: {response}'

    # Test case 2: Empty input
    print('Testing case [2/2] started.')
    try:
        generate_chatbot_response('')
        assert False, 'Test case [2/2] failed: ValueError was not raised for empty input.'
    except ValueError as e:
        assert str(e) == 'Input message cannot be empty.', f'Test case [2/2] failed: Wrong error message. Received: {str(e)}'
    
    print('Testing finished.')

# call_test_function_line --------------------

test_generate_chatbot_response()