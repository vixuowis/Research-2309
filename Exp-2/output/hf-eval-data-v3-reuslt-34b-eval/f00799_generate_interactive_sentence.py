# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_interactive_sentence(masked_sentence: str) -> str:
    """
    Generate an interactive sentence by filling the masked word in the given sentence.

    Args:
        masked_sentence (str): The sentence with a masked word, e.g., 'Tell me more about your [MASK] hobbies.'

    Returns:
        str: The completed sentence with the masked word filled.

    Raises:
        OSError: If there is a problem with the disk quota or the model cannot be loaded.
    """
    unmasker = pipeline("fill-mask")
    return unmasker(masked_sentence)[0]["sequence"]

# test_function_code --------------------

def test_generate_interactive_sentence():
    """
    Test the function generate_interactive_sentence.
    """
    try:
        assert generate_interactive_sentence('Tell me more about your [MASK] hobbies.') is not None
        assert generate_interactive_sentence('I love to [MASK] in my free time.') is not None
        assert generate_interactive_sentence('My favorite food is [MASK].') is not None
        print('All Tests Passed')
    except AssertionError as e:
        print(f'Test failed: {e}')
        raise


# call_test_function_code --------------------

test_generate_interactive_sentence()