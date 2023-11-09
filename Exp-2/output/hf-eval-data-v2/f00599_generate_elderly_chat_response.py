# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_elderly_chat_response(user_question: str) -> str:
    """
    This function generates a conversational response based on the persona of an elderly person.
    It uses the PygmalionAI/pygmalion-2.7b model from Hugging Face Transformers.

    Args:
        user_question (str): The question asked by the user.

    Returns:
        str: The generated response.
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
    This function tests the generate_elderly_chat_response function.
    It uses a sample question and checks if the response is not None.
    """
    sample_question = "What advice would you give to someone just starting their career?"
    response = generate_elderly_chat_response(sample_question)
    assert response is not None, "The function did not return a response."
    assert isinstance(response, str), "The function did not return a string."
    assert '[Old Person]:' in response, "The function did not generate a response in the correct format."

# call_test_function_code --------------------

test_generate_elderly_chat_response()