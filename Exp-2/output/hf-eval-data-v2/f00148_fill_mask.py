# function_import --------------------

from transformers import pipeline

# function_code --------------------

def fill_mask(sentence):
    """
    This function uses the DebertaModel from Hugging Face Transformers to fill in the gaps in a given sentence.
    
    Args:
        sentence (str): The sentence with a '[MASK]' placeholder where the missing word or phrase should be.
    
    Returns:
        str: The sentence with the '[MASK]' placeholder replaced by the predicted word or phrase.
    
    Raises:
        ValueError: If the sentence does not contain a '[MASK]' placeholder.
    """
    if '[MASK]' not in sentence:
        raise ValueError("The sentence does not contain a '[MASK]' placeholder.")
    
    fill_mask_model = pipeline('fill-mask', model='microsoft/deberta-v3-base')
    result = fill_mask_model(sentence)
    return result[0]['sequence']

# test_function_code --------------------

def test_fill_mask():
    """
    This function tests the fill_mask function using a sample sentence.
    """
    sentence = 'The weather today is [MASK] than yesterday.'
    expected_result = 'The weather today is better than yesterday.'
    assert fill_mask(sentence) == expected_result

# call_test_function_code --------------------

test_fill_mask()