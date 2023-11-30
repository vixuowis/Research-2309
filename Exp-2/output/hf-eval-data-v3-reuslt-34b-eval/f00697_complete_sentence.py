# function_import --------------------

from transformers import pipeline

# function_code --------------------

def complete_sentence(sentence: str) -> str:
    """
    Complete the sentence by filling the masked word using the 'roberta-base' model.

    Args:
        sentence (str): The sentence with a masked word represented by <mask>.

    Returns:
        str: The completed sentence.

    Raises:
        OSError: If there is an issue with disk space or permissions.
    """
    fill_mask = pipeline(
        "fill-mask",
        model="roberta-base",
        tokenizer="roberta-base"
    )

    sentence = sentence.replace("\\n", "\n")
    
    result = fill_mask(sentence)
    return result[0]["sequence"]


# test_function_code --------------------

def test_complete_sentence():
    """
    Test the complete_sentence function with various test cases.
    """
    assert complete_sentence('In the story, the antagonist represents the <mask> nature of humanity.')
    assert complete_sentence('The <mask> is the largest animal on earth.')
    assert complete_sentence('The sun is the <mask> of the solar system.')
    return 'All Tests Passed'


# call_test_function_code --------------------

test_complete_sentence()