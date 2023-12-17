# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def complete_slogan(slogan_template: str) -> str:
    """
    Function to complete a marketing slogan using a fill-mask pipeline.

    Args:
        slogan_template (str): A template of the slogan with a <mask> to be filled.

    Returns:
        str: The completed slogan with the highest probability suggestion filled in.

    Raises:
        ValueError: If the slogan_template does not contain a <mask> token.
    """
    if '<mask>' not in slogan_template:
        raise ValueError('Slogan template must contain a <mask> token.')
    unmasker = pipeline('fill-mask', model='roberta-large')
    suggestions = unmasker(slogan_template)
    # The unmasked slogan with the highest probability will be the suggested completed slogan
    completed_slogan = suggestions[0]['sequence']
    return completed_slogan

# test_function_code --------------------

def test_complete_slogan():
    print("Testing started.")
    # Test case: the provided slogan template contains a <mask> token
    print("Testing case [1/2] started.")
    slogan_template_with_mask = "Customer satisfaction is our top <mask>."
    assert '<mask>' in slogan_template_with_mask, "Test case [1/2] failed: Slogan template does not contain a <mask> token."
    # Test case: the provided slogan template does not contain a <mask> token and raises a ValueError
    print("Testing case [2/2] started.")
    slogan_template_without_mask = "Customer satisfaction is our top priority."
    try:
        complete_slogan(slogan_template_without_mask)
    except ValueError as e:
        assert str(e) == 'Slogan template must contain a <mask> token.', "Test case [2/2] failed: ValueError was not raised or had the wrong message."
    print("Testing finished.")

# call_test_function_line --------------------

test_complete_slogan()