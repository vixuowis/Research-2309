# function_import --------------------

from transformers import pipeline

# function_code --------------------

def fill_masked_text(masked_text):
    """
    This function takes a text with masked words and uses the Hugging Face Transformers pipeline to fill the masked words.

    Args:
        masked_text (str): The text with masked words.

    Returns:
        list: A list of dictionaries. Each dictionary contains a filled sentence and the score of the filled word.

    Raises:
        OSError: If there is not enough disk space to download the model.
    """
    mask_unmasker = pipeline('fill-mask', model='xlm-roberta-large')
    filled_sentences = mask_unmasker(masked_text)
    return filled_sentences

# test_function_code --------------------

def test_fill_masked_text():
    """
    This function tests the fill_masked_text function.
    """
    sample_text = "<mask> are large, slow-moving reptiles native to the southeastern United States. They are well-adapted to life in <mask>, and they are a common sight in swamps, rivers, and lakes."
    result = fill_masked_text(sample_text)
    assert isinstance(result, list), 'The result should be a list.'
    assert len(result) > 0, 'The result list should not be empty.'
    for item in result:
        assert 'sequence' in item, 'Each item in the result list should be a dictionary with a sequence key.'
        assert 'score' in item, 'Each item in the result list should be a dictionary with a score key.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_fill_masked_text()