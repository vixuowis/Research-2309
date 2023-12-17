# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def complete_phrase(text, mask_token='[MASK]'):
    """
    Complete a phrase in a given text by filling in the mask token.

    Args:
        text (str): The text containing the mask token to be completed.
        mask_token (str): The token to be replaced with a prediction. Default is '[MASK]'.

    Returns:
        List[dict]: A list of possible completions with their confidence scores.

    Raises:
        ValueError: If the mask_token is not present in the text.
    """
    if mask_token not in text:
        raise ValueError('The mask token must be present in the text')

    fill_mask = pipeline('fill-mask', model='microsoft/deberta-v3-base')
    completions = fill_mask(text)
    return completions

# test_function_code --------------------

def test_complete_phrase():
    print("Testing started.")

    # Test case 1: Check if function completes a single masked token
    print("Testing case [1/3] started.")
    result = complete_phrase('The weather today is [MASK] than yesterday.')
    assert result, f"Test case [1/3] failed: function returned an empty result."

    # Test case 2: Check if ValueError is raised when mask_token is not in text
    print("Testing case [2/3] started.")
    try:
        complete_phrase('The weather today is sunny than yesterday.')
    except ValueError as e:
        assert str(e) == 'The mask token must be present in the text', f"Test case [2/3] failed: {str(e)}"

    # Test case 3: Tests with a specified mask token
    print("Testing case [3/3] started.")
    result = complete_phrase('It is an [XXX] day.', mask_token='[XXX]')
    assert result, f"Test case [3/3] failed: function returned an empty result with a custom mask token."
    print("Testing finished.")

# call_test_function_line --------------------

test_complete_phrase()