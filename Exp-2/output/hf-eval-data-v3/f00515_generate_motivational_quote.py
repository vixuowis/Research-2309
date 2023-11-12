# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_motivational_quote(prompt: str, max_length: int = 50) -> str:
    '''
    Generate a motivational quote related to sports using a text generation model.

    Args:
        prompt (str): The initial text to prompt the model.
        max_length (int, optional): The maximum length of the generated text. Defaults to 50.

    Returns:
        str: The generated motivational quote.
    '''
    text_generator = pipeline('text-generation', model='TehVenom/PPO_Pygway-V8p4_Dev-6b')
    generated_text = text_generator(prompt, max_length=max_length)[0]['generated_text']
    return generated_text

# test_function_code --------------------

def test_generate_motivational_quote():
    '''
    Test the function generate_motivational_quote.
    '''
    quote1 = generate_motivational_quote('Motivational quote about sports:')
    assert isinstance(quote1, str) and len(quote1) > 0

    quote2 = generate_motivational_quote('Create a motivational sports quote:')
    assert isinstance(quote2, str) and len(quote2) > 0

    quote3 = generate_motivational_quote('Inspire me with a sports quote:', max_length=100)
    assert isinstance(quote3, str) and len(quote3) > 0

    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_motivational_quote()