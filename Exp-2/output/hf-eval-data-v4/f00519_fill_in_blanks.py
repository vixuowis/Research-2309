# requirements_file --------------------

!pip install -U transformers 

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def fill_in_blanks(sentence_with_mask):
    """
    Fills in the blanks denoted by [MASK] in the given sentence using DeBERTa model.
    :param sentence_with_mask: The sentence with [MASK] token where the blank should be filled.
    :return: A list of dictionaries containing the possible words with their scores that can fill the blank.
    """
    fill_mask = pipeline('fill-mask', model='microsoft/deberta-base')
    result = fill_mask(sentence_with_mask)
    return result

# test_function_code --------------------

def test_fill_in_blanks():
    print("Testing started.")

    # Test case 1: Regular sentence
    print("Testing case [1/1] started.")
    sentence = 'The capital of France is [MASK].'
    result = fill_in_blanks(sentence)
    assert result and type(result) is list and result[0]['sequence'].strip() == 'The capital of France is Paris.', f"Test case [1/1] failed: {result}"
    print("Test case [1/1] passed.")
    print("Testing finished.")

test_fill_in_blanks()