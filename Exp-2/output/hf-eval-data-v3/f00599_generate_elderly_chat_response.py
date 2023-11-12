# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_elderly_chat_response(user_question: str) -> str:
    """
    Generate a chat response based on the persona of an elderly person using the PygmalionAI/pygmalion-2.7b model.

    Args:
        user_question (str): The user's question to the elderly persona.

    Returns:
        str: The generated response from the elderly persona.

    Raises:
        OSError: If there is an issue with disk space when downloading the model.
    """
    generated_pipeline = pipeline('text-generation', model='PygmalionAI/pygmalion-2.7b')
    persona = "Old Person's Persona: I am an elderly person with a lot of life experience and wisdom. I enjoy sharing stories and offering advice to younger generations."
    history = "<START>"
    input_prompt = f"{persona}{history}{user_question}[Old Person]:"
    response = generated_pipeline(input_prompt)
    return response[0]['generated_text']

# test_function_code --------------------

def test_generate_elderly_chat_response():
    """
    Test the generate_elderly_chat_response function.
    """
    sample_question1 = "You: What advice would you give to someone just starting their career?"
    sample_question2 = "You: What is your favorite memory?"
    sample_question3 = "You: How has the world changed since you were young?"
    assert isinstance(generate_elderly_chat_response(sample_question1), str)
    assert isinstance(generate_elderly_chat_response(sample_question2), str)
    assert isinstance(generate_elderly_chat_response(sample_question3), str)
    print('All Tests Passed')

# call_test_function_code --------------------

test_generate_elderly_chat_response()