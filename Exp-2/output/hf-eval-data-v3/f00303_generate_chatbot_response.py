# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_chatbot_response(input_prompt: str) -> str:
    '''
    Generate a response from a chatbot using the PygmalionAI/pygmalion-1.3b model.

    Args:
        input_prompt (str): The input prompt for the chatbot, which includes character persona, dialogue history, and the user input message.

    Returns:
        str: The generated response from the chatbot.
    '''
    chatbot = pipeline('text-generation', 'PygmalionAI/pygmalion-1.3b')
    response = chatbot(input_prompt)
    return response

# test_function_code --------------------

def test_generate_chatbot_response():
    '''
    Test the generate_chatbot_response function.
    '''
    character_persona = 'CompanyBot\'s Persona: I am a helpful chatbot designed to answer questions about our products and services.'
    dialogue_history = ''
    input_prompt = (
        f'{character_persona}\n'
        f'{dialogue_history}'
        'You: What products do you offer?\n'
    )
    response = generate_chatbot_response(input_prompt)
    assert isinstance(response, str), 'The response should be a string.'
    assert len(response) > 0, 'The response should not be empty.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_chatbot_response()