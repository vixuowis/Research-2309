# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def complete_french_sentence(sentence_with_mask):
    """
    Complete a sentence with a missing word in French using the CamemBERT base model.

    Args:
        sentence_with_mask (str): A sentence in French with '<mask>' token where the word is missing.

    Returns:
        str: The completed sentence with the most likely word filled in the place of '<mask>'.

    Raises:
        ValueError: If the input sentence does not contain '<mask>' token.
    """
    if '<mask>' not in sentence_with_mask:
        raise ValueError("Input sentence must contain '<mask>' token for the missing word.")
    camembert_fill_mask = pipeline('fill-mask', model='camembert-base', tokenizer='camembert-base')
    results = camembert_fill_mask(sentence_with_mask)
    return results[0]['sequence']

# test_function_code --------------------

def test_complete_french_sentence():
    print("Testing started.")

    # Testing case 1: Sentence with one mask
    print("Testing case [1/2] started.")
    sentence_1 = "Le camembert est <mask>."
    result_1 = complete_french_sentence(sentence_1)
    assert '<mask>' not in result_1, f"Test case [1/2] failed: Expected '<mask>' to be replaced, got {result_1}"

    # Testing case 2: Raise ValueError when no mask is present
    print("Testing case [2/2] started.")
    sentence_2 = "Le camembert est délicieux."
    try:
        complete_french_sentence(sentence_2)
        assert False, "Test case [2/2] failed: ValueError not raised when '<mask>' is missing."
    except ValueError as e:
        assert str(e) == "Input sentence must contain '<mask>' token for the missing word.", f"Test case [2/2] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_complete_french_sentence()