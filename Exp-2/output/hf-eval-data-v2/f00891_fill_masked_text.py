# function_import --------------------

from transformers import pipeline

# function_code --------------------

def fill_masked_text(masked_text):
    """
    This function uses the xlm-roberta-large model from Hugging Face Transformers to fill masked words in a given text.

    Args:
        masked_text (str): The text with masked words to be filled.

    Returns:
        list: A list of dictionaries. Each dictionary contains a filled sentence and the score of the prediction.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(masked_text, str):
        raise ValueError('Input must be a string')
    mask_unmasker = pipeline('fill-mask', model='xlm-roberta-large')
    filled_sentence = mask_unmasker(masked_text)
    return filled_sentence

# test_function_code --------------------

def test_fill_masked_text():
    """
    This function tests the fill_masked_text function with a sample text.
    """
    sample_text = "<mask> are large, slow-moving reptiles native to the southeastern United States. They are well-adapted to life in <mask>, and they are a common sight in swamps, rivers, and lakes."
    result = fill_masked_text(sample_text)
    assert isinstance(result, list), 'The result should be a list.'
    assert 'sequence' in result[0], 'Each item in the list should be a dictionary with a sequence key.'
    assert 'score' in result[0], 'Each item in the list should be a dictionary with a score key.'

# call_test_function_code --------------------

test_fill_masked_text()