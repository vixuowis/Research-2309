# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def fill_sentence_blanks(text):
    """
    Fill in the blanks within a sentence denoted by [MASK] tokens.

    Args:
        text (str): The input text containing [MASK] tokens where blanks are to be filled.
    
    Returns:
        list: A list of dictionaries with possible fill-ins for each [MASK] token.

    Raises:
        ValueError: If 'text' is not a string or is empty.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("Input text must be a non-empty string.")
    fill_mask = pipeline('fill-mask', model='microsoft/deberta-base')
    return fill_mask(text)

# test_function_code --------------------

def test_fill_sentence_blanks():
    print("Testing started.")
    
    # Test case 1: A single mask token
    print("Testing case [1/3] started.")
    sentence_1 = "The capital of France is [MASK]."
    result_1 = fill_sentence_blanks(sentence_1)
    assert result_1, f"Test case [1/3] failed: Expected non-empty result, got {result_1}"

    # Test case 2: No mask token
    print("Testing case [2/3] started.")
    sentence_2 = "This is a complete sentence without a mask."
    result_2 = fill_sentence_blanks(sentence_2)
    assert result_2, f"Test case [2/3] failed: Expected non-empty result for sentences without mask, got {result_2}"

    # Test case 3: Invalid input
    print("Testing case [3/3] started.")
    sentence_3 = ""
    try:
        fill_sentence_blanks(sentence_3)
        assert False, "Test case [3/3] failed: Expected ValueError for empty input."
    except ValueError as e:
        assert str(e) == "Input text must be a non-empty string.", f"Test case [3/3] failed: {e}"
    
    print("Testing finished.")

# call_test_function_line --------------------

test_fill_sentence_blanks()