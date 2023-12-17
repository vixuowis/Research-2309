# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_nature_poem(prompt):
    """
    Generates a poem about nature based on the provided prompt.

    Args:
        prompt (str): A string prompt to guide the text generation.

    Returns:
        str: The generated poem as a string.

    Raises:
        ValueError: If prompt is not a string.
    """
    if not isinstance(prompt, str):
        raise ValueError('The prompt must be a string.')

    nlp = pipeline('text-generation', model='sshleifer/tiny-gpt2')
    result = nlp(prompt, max_length=100, num_return_sequences=1)
    return result[0]['generated_text']

# test_function_code --------------------

def test_generate_nature_poem():
    print("Testing started.")
    test_prompt = 'Once upon a time, in a land of greenery and beauty,'

    # Test case 1: Check if the function returns a string.
    print("Testing case [1/1] started.")
    poem = generate_nature_poem(test_prompt)
    assert isinstance(poem, str), f"Test case [1/1] failed: Expected a string, got {type(poem)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_nature_poem()