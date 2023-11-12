# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_poem(prompt: str) -> str:
    '''
    Generate a poem based on a given prompt using the TinyGPT2 model.

    Args:
        prompt (str): The initial string to base the poem on.

    Returns:
        str: The generated poem.
    '''
    nlp = pipeline('text-generation', model='sshleifer/tiny-gpt2')
    result = nlp(prompt)
    return result[0]['generated_text']

# test_function_code --------------------

def test_generate_poem():
    '''
    Test the generate_poem function.
    '''
    assert isinstance(generate_poem('Once upon a time'), str)
    assert isinstance(generate_poem('In a land far away'), str)
    assert isinstance(generate_poem('Under the bright sun'), str)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_poem()