# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_interactive_sentence(masked_sentence):
    """
    Generate an interactive sentence by filling the masked word in the given sentence.

    Args:
        masked_sentence (str): The sentence with a masked word, represented by [MASK].

    Returns:
        str: The completed sentence with the masked word filled.

    Raises:
        ValueError: If the input is not a string or if it doesn't contain a masked word.
    """
    if not isinstance(masked_sentence, str) or '[MASK]' not in masked_sentence:
        raise ValueError('Input should be a string containing a masked word represented by [MASK].')

    unmasker = pipeline('fill-mask', model='albert-base-v2')
    completed_sentence = unmasker(masked_sentence)
    return completed_sentence[0]['sequence']

# test_function_code --------------------

def test_generate_interactive_sentence():
    """
    Test the function generate_interactive_sentence.

    The function is tested with a sentence containing a masked word. The output is checked for being a string and containing the original sentence without the masked word.
    """
    masked_sentence = 'Tell me more about your [MASK] hobbies.'
    completed_sentence = generate_interactive_sentence(masked_sentence)
    assert isinstance(completed_sentence, str), 'The output should be a string.'
    assert '[MASK]' not in completed_sentence, 'The output should not contain the masked word.'
    assert 'Tell me more about your' in completed_sentence and 'hobbies.' in completed_sentence, 'The output should contain the original sentence without the masked word.'

# call_test_function_code --------------------

test_generate_interactive_sentence()