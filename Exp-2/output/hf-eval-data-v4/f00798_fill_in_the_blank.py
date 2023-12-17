# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def fill_in_the_blank(text):
    unmasker = pipeline('fill-mask', model='roberta-base')
    result = unmasker(text)
    predicted_word = result[0]['token_str']
    completed_text = text.replace('<mask>', predicted_word)
    return completed_text

# test_function_code --------------------

def test_fill_in_the_blank():
    print("Testing started.")

    # Test case 1: Check if the function is correctly replacing a mask token with a predicted word
    test_text = "The weather was so <mask> that everyone stayed indoors."
    expected_result = "The weather was so bad that everyone stayed indoors."
    print("Testing case [1/1] started.")
    result = fill_in_the_blank(test_text)
    assert result == expected_result, f"Test case [1/1] failed: Expected {{expected_result}} but got {{result}}"
    print("Test case [1/1] passed.")
    print("Testing finished.")