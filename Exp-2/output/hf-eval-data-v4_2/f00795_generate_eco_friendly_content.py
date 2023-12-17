# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline, set_seed

# function_code --------------------

def generate_eco_friendly_content(prompt: str) -> str:
    """
    Generates marketing content for eco-friendly kitchenware based on a given prompt.

    Args:
        prompt (str): The initial text to inspire the content generation.

    Returns:
        str: The generated marketing content.

    Raises:
        ValueError: If the prompt is empty or None.
    """
    if not prompt:
        raise ValueError('The prompt cannot be empty or None.')

    set_seed(42)
    generator = pipeline('text-generation', model='facebook/opt-125m')
    generated_content = generator(prompt, max_length=100, do_sample=True)[0]['generated_text']
    return generated_content

# test_function_code --------------------

def test_generate_eco_friendly_content():
    print('Testing started.')

    # Test case 1: Non-empty prompt
    print('Testing case [1/2] started.')
    prompt = 'Discover our sustainable kitchenware:'
    content = generate_eco_friendly_content(prompt)
    assert content.startswith(prompt), f'Test case [1/2] failed: Content does not start with the prompt.'

    # Test case 2: Empty prompt
    print('Testing case [2/2] started.')
    prompt = ''
    try:
        _ = generate_eco_friendly_content(prompt)
        assert False, 'Test case [2/2] failed: ValueError not raised for empty prompt.'
    except ValueError as e:
        assert str(e) == 'The prompt cannot be empty or None.', f'Test case [2/2] failed: {e}'

    print('Testing finished.')

# call_test_function_line --------------------

test_generate_eco_friendly_content()