# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def complete_missing_words(masked_sentence: str) -> str:
    """
    Complete missing words in a sentence with a mask token using ALBERT model.

    Args:
        masked_sentence (str): The sentence with [MASK] token for ALBERT to predict.

    Returns:
        str: The completed sentence(s) with the masked word(s) predicted by the model.

    Raises:
        ValueError: If masked_sentence is not a string or if it does not contain any [MASK] token.
    """
    if not isinstance(masked_sentence, str) or '[MASK]' not in masked_sentence:
        raise ValueError('The input must be a string containing at least one [MASK] token.')
    unmasker = pipeline('fill-mask', model='albert-base-v2')
    predictions = unmasker(masked_sentence)
    completed_sentences = [prediction['sequence'] for prediction in predictions]
    return completed_sentences[0] if completed_sentences else ''

# test_function_code --------------------

def test_complete_missing_words():
    print("Testing started.")

    test_cases = [
        ('Tell me about your [MASK] hobbies.', 'Tell me about your favorite hobbies.'),
        ('I enjoy [MASK] on the weekends.', 'I enjoy hiking on the weekends.'),
        ('We could go out for a [MASK] evening.', 'We could go out for a romantic evening.')
    ]

    for idx, (input_sentence, expected_output) in enumerate(test_cases, start=1):
        print(f"Testing case [{idx}/{len(test_cases)}] started.")
        result = complete_missing_words(input_sentence)
        assert result == expected_output, f"Test case [{idx}/{len(test_cases)}] failed: expected {expected_output}, got {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_complete_missing_words()