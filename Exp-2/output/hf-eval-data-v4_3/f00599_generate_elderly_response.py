# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_elderly_response(user_question, persona='<Old Person\'s Persona: I am an elderly person with a lot of life experience and wisdom. I enjoy sharing stories and offering advice to younger generations.>', history='<START>'):
    """Generate a response from an elderly person's perspective.

    Args:
        user_question (str): The question asked by the user.
        persona (str): The persona of the elderly character.
        history (str): The history of the conversation with the elderly character.

    Returns:
        str: The generated response text.

    Raises:
        ValueError: If the user_question is empty.
    """
    if not user_question:
        raise ValueError('The user_question cannot be empty.')

    generated_pipeline = pipeline('text-generation', model='PygmalionAI/pygmalion-2.7b')
    input_prompt = f"{persona}{history}{user_question}[Old Person]:"
    response = generated_pipeline(input_prompt)
    return response[0]['generated_text']

# test_function_code --------------------

def test_generate_elderly_response():
    print("Testing started.")

    # Test case 1: Typical question
    print("Testing case [1/3] started.")
    question = "You: What advice would you give to someone just starting their career?"
    response = generate_elderly_response(question)
    assert isinstance(response, str), f"Test case [1/3] failed: Response is not a string."

    # Test case 2: Empty question
    print("Testing case [2/3] started.")
    try:
        generate_elderly_response('')
    except ValueError as e:
        assert str(e) == 'The user_question cannot be empty.', f"Test case [2/3] failed: {e}"

    # Test case 3: Check response contains persona
    print("Testing case [3/3] started.")
    persona = '<Old Person\'s Persona: I like to reminisce about the past and tell stories.>'
    response = generate_elderly_response(question, persona=persona)
    assert persona in response, f"Test case [3/3] failed: Persona not included in response."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_elderly_response()