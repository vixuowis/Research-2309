# function_import --------------------

from transformers import pipeline, set_seed

# function_code --------------------

def generate_marketing_content(prompt: str) -> str:
    """
    Generate marketing content using the OPT pre-trained transformer 'facebook/opt-125m'.

    Args:
        prompt (str): The initial prompt to feed to the text generation model.

    Returns:
        str: The generated marketing content.

    Raises:
        OSError: If there is a problem with the disk quota.
    """
    set_seed(42)
    generator = pipeline('text-generation', model='facebook/opt-125m')
    generated_content = generator(prompt, max_length=100, do_sample=True)[0]['generated_text']
    return generated_content

# test_function_code --------------------

def test_generate_marketing_content():
    """
    Test the generate_marketing_content function.
    """
    prompt = 'Introducing our new line of eco-friendly kitchenware:'
    generated_content = generate_marketing_content(prompt)
    assert isinstance(generated_content, str)
    assert len(generated_content) > 0
    print('All Tests Passed')

# call_test_function_code --------------------

if __name__ == '__main__':
    test_generate_marketing_content()