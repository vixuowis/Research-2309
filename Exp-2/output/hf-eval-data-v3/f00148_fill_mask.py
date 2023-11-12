# function_import --------------------

from transformers import pipeline

# function_code --------------------

def fill_mask(sentence: str) -> str:
    '''
    This function uses the DebertaModel from Hugging Face Transformers to fill in the gaps in a sentence.

    Args:
        sentence (str): The sentence with a '[MASK]' placeholder where the missing word or phrase should be.

    Returns:
        str: The sentence with the '[MASK]' placeholder replaced by the predicted word or phrase.
    '''
    fill_mask = pipeline('fill-mask', model='microsoft/deberta-v3-base')
    result = fill_mask(sentence)
    return result[0]['sequence']

# test_function_code --------------------

def test_fill_mask():
    '''
    This function tests the fill_mask function.
    '''
    assert fill_mask('The weather today is [MASK] than yesterday.') == 'The weather today is better than yesterday.'
    assert fill_mask('Hugging Face is a [MASK] company.') == 'Hugging Face is a great company.'
    assert fill_mask('I [MASK] to the store to buy some groceries.') == 'I went to the store to buy some groceries.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_fill_mask()