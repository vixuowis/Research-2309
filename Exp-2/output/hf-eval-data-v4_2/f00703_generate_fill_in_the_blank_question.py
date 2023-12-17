# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import DebertaV2Tokenizer, DebertaV2ForMaskedLM

# function_code --------------------

def generate_fill_in_the_blank_question(sentence: str, mask_position: int) -> str:
    """
    Generates a fill-in-the-blank question by masking a word at a given position in a sentence.

    Args:
        sentence (str): The sentence from which to create the fill-in-the-blank question.
        mask_position (int): The position of the word to mask (zero-indexed).

    Returns:
        str: A sentence with the word at the given position replaced by '[MASK]'.
    
    Raises:
        ValueError: If mask_position is out of the range of the sentence word count.
    """
    words = sentence.split()
    if not 0 <= mask_position < len(words):
        raise ValueError('mask_position is out of the range of the sentence word count.')
    words[mask_position] = '[MASK]'
    return ' '.join(words)

# test_function_code --------------------

def test_generate_fill_in_the_blank_question():
    print("Testing started.")
    sentence = "The cat chased the mouse and then climbed the tree."

    # Testing case 1: masking the fifth word (mouse)
    print("Testing case [1/3] started.")
    expected_output = "The cat chased the [MASK] and then climbed the tree."
    assert generate_fill_in_the_blank_question(sentence, 4) == expected_output, f"Test case [1/3] failed: expected {expected_output}"

    # Testing case 2: masking the last word (tree)
    print("Testing case [2/3] started.")
    expected_output = "The cat chased the mouse and then climbed the [MASK]."
    assert generate_fill_in_the_blank_question(sentence, -1) == expected_output, f"Test case [2/3] failed: expected {expected_output}"

    # Testing case 3: Invalid mask_position
    print("Testing case [3/3] started.")
    try:
        generate_fill_in_the_blank_question(sentence, 10)
        assert False, "Test case [3/3] failed: ValueError was not raised for invalid mask_position"
    except ValueError as e:
        assert str(e) == 'mask_position is out of the range of the sentence word count.', f"Test case [3/3] failed: {str(e)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_fill_in_the_blank_question()