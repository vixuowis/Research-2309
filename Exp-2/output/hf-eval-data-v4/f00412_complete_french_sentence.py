# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def complete_french_sentence(sentence_with_mask):
    """
    Complete a French sentence with a masked token using the camembert-base model.

    Args:
        sentence_with_mask (str): A French sentence with a '<mask>' token representing
                                 the position where a word is missing.

    Returns:
        list: A list of dictionaries containing the possible completions with their scores.
    """
    camembert_fill_mask = pipeline('fill-mask', model='camembert-base', tokenizer='camembert-base')
    results = camembert_fill_mask(sentence_with_mask)
    return results

# test_function_code --------------------

def test_complete_french_sentence():
    print("Testing complete_french_sentence function.")

    # Test case 1: Single mask token
    sentence_with_mask = 'Le camembert est <mask> :)'
    results = complete_french_sentence(sentence_with_mask)
    assert len(results) > 0, "Test case 1 failed: No completions found."

    # Test case 2: No mask token in sentence
    sentence_without_mask = 'Le camembert est délicieux :)'
    results = complete_french_sentence(sentence_without_mask)
    assert len(results) == 0, "Test case 2 failed: Completions found for sentence without mask."

    # Test case 3: Mask token at the beginning of sentence
    sentence_with_mask_at_start = '<mask> est un fromage français.'
    results = complete_french_sentence(sentence_with_mask_at_start)
    assert len(results) > 0, "Test case 3 failed: No completions found for sentence with mask at start."

    print("Testing complete_french_sentence function finished successfully.")

# Run the test function
test_complete_french_sentence()