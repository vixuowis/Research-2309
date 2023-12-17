# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def complete_english_essay_sentence(sentence: str) -> str:
    """
    Completes an English sentence by filling in the masked word using a pre-trained
    language model.

    Args:
        sentence (str): A sentence with a '<mask>' token where a word should be predicted.

    Returns:
        str: The sentence with the '<mask>' token replaced by the predicted word.

    Raises:
        ValueError: If the input sentence does not contain the '<mask>' token.
    """
    if '<mask>' not in sentence:
        raise ValueError("Input sentence must contain the '<mask>' token.")
    unmasker = pipeline('fill-mask', model='roberta-base')
    result = unmasker(sentence)
    completed_sentence = result[0]['sequence']
    return completed_sentence

# test_function_code --------------------

def test_complete_english_essay_sentence():
    print("Testing started.")

    # Test case 1: Sentence with one masked token
    print("Testing case [1/3] started.")
    sentence_1 = "In the story, the antagonist represents the <mask> nature of humanity."
    expected_1 = "In the story, the antagonist represents the complex nature of humanity."
    result_1 = complete_english_essay_sentence(sentence_1)
    assert result_1 == expected_1, f"Test case [1/3] failed: {result_1} != {expected_1}"

    # Test case 2: Sentence with no masked token should raise an exception
    print("Testing case [2/3] started.")
    sentence_2 = "In the story, the antagonist represents the dark nature of humanity."
    try:
        result_2 = complete_english_essay_sentence(sentence_2)
        assert False, "Test case [2/3] failed: ValueError not raised."
    except ValueError as e:
        assert str(e) == "Input sentence must contain the '<mask>' token.", f"Test case [2/3] failed: Wrong exception message: {e}"

    # Test case 3: Sentence with multiple masked tokens
    print("Testing case [3/3] started.")
    sentence_3 = "The hero stood up to the <mask> forces of the <mask> enemy."
    result_3 = complete_english_essay_sentence(sentence_3)
    assert '<mask>' no

# call_test_function_line --------------------

test_complete_english_essay_sentence()