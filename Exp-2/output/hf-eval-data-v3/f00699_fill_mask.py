# function_import --------------------

from transformers import pipeline

# function_code --------------------

def fill_mask(sentence: str) -> str:
    """
    This function fills the missing word in a sentence using a multilingual model.

    Args:
        sentence (str): The sentence with a missing word, represented by [MASK].

    Returns:
        str: The sentence with the missing word filled.
    """
    unmasker = pipeline('fill-mask', model='distilbert-base-multilingual-cased')
    result = unmasker(sentence)
    return result[0]['sequence']

# test_function_code --------------------

def test_fill_mask():
    """
    This function tests the fill_mask function with some test cases.
    """
    assert fill_mask('Hello, I am a [MASK] model.') == 'Hello, I am a language model.'
    assert fill_mask('This is a [MASK] test.') == 'This is a simple test.'
    assert fill_mask('The weather is [MASK] today.') == 'The weather is nice today.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_fill_mask()