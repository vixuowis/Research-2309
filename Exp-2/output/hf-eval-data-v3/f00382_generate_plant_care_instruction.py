# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_plant_care_instruction(prompt: str) -> str:
    """
    Generate a short and simple plant care instruction based on the provided prompt.

    Args:
        prompt (str): The input prompt for the text generation.

    Returns:
        str: The generated plant care instruction.
    """
    # Initialize the pipeline with 'text-generation' as task and the pre-trained GPT model
    text_generator = pipeline('text-generation', model='gpt2')
    # Generate the text by providing the input prompt
    generated_text = text_generator(prompt)[0]['generated_text']
    return generated_text

# test_function_code --------------------

def test_generate_plant_care_instruction():
    """
    Test the function generate_plant_care_instruction.
    """
    # Test case 1: Check the type of the output
    assert isinstance(generate_plant_care_instruction('I want to give a potted plant to my friend.'), str)
    # Test case 2: Check the output with a specific prompt
    assert 'water' in generate_plant_care_instruction('How to care for a succulent?').lower()
    # Test case 3: Check the output with another specific prompt
    assert 'light' in generate_plant_care_instruction('How to care for a fern?').lower()
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_plant_care_instruction()