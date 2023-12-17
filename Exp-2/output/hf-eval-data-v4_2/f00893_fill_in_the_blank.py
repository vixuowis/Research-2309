# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def fill_in_the_blank(sentence):
    """
    Fill in the blank space in a sentence using BERT large model.

    Args:
        sentence (str): The sentence with a [MASK] token where the blank should be filled.

    Returns:
        str: A sentence with the [MASK] token replaced by the predicted word.

    Raises:
        ValueError: If the sentence does not contain a [MASK] token.
    """
    if '[MASK]' not in sentence:
        raise ValueError('The sentence must contain a [MASK] token.')

    fill_mask = pipeline('fill-mask', model='bert-large-uncased')
    result = fill_mask(sentence)

    return result[0]['sequence']

# test_function_code --------------------

def test_fill_in_the_blank():
    print("Testing started.")
    test_sentences = [
        'The capital of France is [MASK].',
        'Hugging Face creates state-of-the-art [MASK] models.',
        'The quick brown fox jumps over the lazy [MASK].'
    ]
    expected_results = [
        'the capital of france is paris.',
        'hugging face creates state-of-the-art machine learning models.',
        'the quick brown fox jumps over the lazy dog.'
    ]
    
    for i, (sentence, expected) in enumerate(zip(test_sentences, expected_results), 1):
        print(f"Testing case [{i}/{len(test_sentences)}] started.")
        filled_sentence = fill_in_the_blank(sentence)
        assert filled_sentence.lower() == expected, f"Test case [{i}/{len(test_sentences)}] failed: Expected '{{expected}}', got '{{filled_sentence}}'"
    print("Testing finished.")

# call_test_function_line --------------------

test_fill_in_the_blank()