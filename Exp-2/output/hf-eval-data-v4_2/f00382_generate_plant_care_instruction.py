# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_plant_care_instruction(prompt: str) -> str:
    """
    Generate a short and simple plant care instruction based on the provided prompt.

    Args:
        prompt (str): The input prompt describing the type of plant.

    Returns:
        str: The generated plant care instruction.

    Raises:
        ValueError: If the input type is not a string.
    """
    if not isinstance(prompt, str):
        raise ValueError('Input prompt must be a string')
    text_generator = pipeline('text-generation', model='gpt2')
    results = text_generator(prompt, max_length=50, clean_up_tokenization_spaces=True)
    instruction = results[0]['generated_text']
    return instruction.strip()

# test_function_code --------------------

def test_generate_plant_care_instruction():
    print("Testing started.")
    # Test case 1: Simple prompt
    print("Testing case [1/3] started.")
    result = generate_plant_care_instruction('Care instructions for a succulent plant:')
    assert isinstance(result, str), f"Test case [1/3] failed: Expected string, got {type(result)}"

    # Test case 2: Non-string input
    print("Testing case [2/3] started.")
    try:
        generate_plant_care_instruction(None)
    except ValueError as e:
        assert str(e) == 'Input prompt must be a string', f"Test case [2/3] failed: {e}"

    # Test case 3: Long prompt
    print("Testing case [3/3] started.")
    long_prompt = 'Instructions for taking care of a very rare and delicate tropical orchid:'
    result = generate_plant_care_instruction(long_prompt)
    assert isinstance(result, str), f"Test case [3/3] failed: Expected string, got {type(result)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_plant_care_instruction()