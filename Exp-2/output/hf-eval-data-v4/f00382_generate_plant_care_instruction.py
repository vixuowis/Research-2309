# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_plant_care_instruction(prompt):
    """
    Generate a short plant care instruction based on the given prompt.

    Parameters:
        prompt (str): The description or name of the plant.

    Returns:
        str: The generated plant care instruction.
    """
    generator = pipeline('text-generation', model='gpt2')
    instructions = generator(prompt, max_length=50, num_return_sequences=1)
    return instructions[0]['generated_text']

# test_function_code --------------------

def test_generate_plant_care_instruction():
    print("Testing generate_plant_care_instruction function.")

    # Test case 1: Check if the instruction is returned
    instruction = generate_plant_care_instruction('care for a Spider Plant')
    assert type(instruction) == str, "Test case failed: The function did not return a string."
    assert len(instruction) > 0, "Test case failed: No instruction generated."

    print("All tests passed for generate_plant_care_instruction function.")

# Run the test function
test_generate_plant_care_instruction()