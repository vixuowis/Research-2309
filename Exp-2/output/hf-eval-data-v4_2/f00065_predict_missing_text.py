# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def predict_missing_text(sentence: str) -> list:
    """
    Predicts the most plausible text to fill the missing part in a given sentence.

    Args:
        sentence (str): The sentence with a missing part indicated by [MASK].

    Returns:
        list: A list of dictionaries with predictions and their corresponding scores.

    Raises:
        ValueError: If 'sentence' does not contain [MASK] token.
    """
    if '[MASK]' not in sentence:
        raise ValueError("The input sentence must contain the [MASK] token.")
    unmasker = pipeline('fill-mask', model='albert-base-v2')
    return unmasker(sentence)

# test_function_code --------------------

def test_predict_missing_text():
    print("Testing started.")

    # Testing case 1: Single [MASK] token
    print("Testing case [1/2] started.")
    filled_sentence = predict_missing_text("Hello I'm a [MASK] model.")
    assert type(filled_sentence) is list, f"Test case [1/2] failed: Expected a list but got {type(filled_sentence)}"
    assert len(filled_sentence) > 0, "Test case [1/2] failed: No predictions found"

    # Testing case 2: Verify ValueError on missing [MASK]
    print("Testing case [2/2] started.")
    try:
        predict_missing_text("Hello I'm a language model.")
        assert False, "Test case [2/2] failed: ValueError was not raised."
    except ValueError:
        pass  # If ValueError is raised, the test passes

    print("Testing finished.")

# call_test_function_line --------------------

test_predict_missing_text()