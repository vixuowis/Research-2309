# function_import --------------------

from transformers import pipeline

# function_code --------------------

def fill_mask(text: str) -> str:
    '''
    Fill in the blanks in a given text using a language model.

    Args:
        text (str): The text with a blank denoted by [MASK].

    Returns:
        str: The text with the blank filled in.
    '''
    fill_mask_model = pipeline('fill-mask', model='microsoft/deberta-base')
    result = fill_mask_model(text)
    return result

# test_function_code --------------------

def test_fill_mask():
    '''
    Test the fill_mask function.
    '''
    assert fill_mask('The capital of France is [MASK].')[0]['sequence'] == 'The capital of France is Paris.'
    assert fill_mask('The [MASK] is the largest planet in the solar system.')[0]['sequence'] == 'The Jupiter is the largest planet in the solar system.'
    assert fill_mask('The [MASK] is the smallest planet in the solar system.')[0]['sequence'] == 'The Mercury is the smallest planet in the solar system.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_fill_mask()