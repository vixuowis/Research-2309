# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline, set_seed

# function_code --------------------

def generate_eco_friendly_kitchenware_content(prompt):
    """
    Generate marketing content for eco-friendly kitchenware.

    Parameters:
        prompt (str): The initial prompt to feed into the text generation model.

    Returns:
        str: Generated marketing content based on the initial prompt.
    """
    set_seed(42)
    generator = pipeline('text-generation', model='facebook/opt-125m')
    generated_content = generator(prompt, max_length=100, do_sample=True)[0]['generated_text']
    return generated_content


# test_function_code --------------------

def test_generate_eco_friendly_kitchenware_content():
    print("Testing generate_eco_friendly_kitchenware_content function.")
    prompt = "Introducing our new line of eco-friendly kitchenware:"

    # Test case 1: Check if the function returns a string
    print("Testing case [1/1] started.")
    content = generate_eco_friendly_kitchenware_content(prompt)
    assert isinstance(content, str), "Test case [1/1] failed: The output should be a string."
    print("Testing finished.")

# Run the test function
test_generate_eco_friendly_kitchenware_content()
