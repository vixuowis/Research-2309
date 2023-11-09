from transformers import pipeline


def generate_interactive_sentence(masked_sentence):
    """
    Generate an interactive sentence by filling the masked word in the given sentence.

    Args:
        masked_sentence (str): The sentence with a masked word, e.g., 'Tell me more about your [MASK] hobbies.'

    Returns:
        str: The completed sentence with the masked word filled.

    Raises:
        ValueError: If the input is not a string or if it doesn't contain a masked word.
    """
    if not isinstance(masked_sentence, str) or '[MASK]' not in masked_sentence:
        raise ValueError('Input should be a string containing a masked word.')

    unmasker = pipeline('fill-mask', model='albert-base-v2')
    completed_sentence = unmasker(masked_sentence)

    return completed_sentence[0]['sequence']