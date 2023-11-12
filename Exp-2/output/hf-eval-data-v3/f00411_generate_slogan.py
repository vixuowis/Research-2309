# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_slogan(slogan_mask: str) -> str:
    '''
    Generate a marketing slogan by filling in the masked portion of the input text.

    Args:
        slogan_mask (str): The input text with a mask, e.g., "Customer satisfaction is our top <mask>."

    Returns:
        str: The completed slogan with the highest probability.
    '''
    unmasker = pipeline('fill-mask', model='roberta-large')
    suggestions = unmasker(slogan_mask)
    completed_slogan = suggestions[0]['sequence']
    return completed_slogan

# test_function_code --------------------

def test_generate_slogan():
    '''
    Test the generate_slogan function.
    '''
    assert generate_slogan("Customer satisfaction is our top <mask>.") != ""
    assert generate_slogan("Our products are <mask>.") != ""
    assert generate_slogan("We strive for <mask>.") != ""
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_slogan()