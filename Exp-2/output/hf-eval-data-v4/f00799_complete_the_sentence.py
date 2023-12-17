# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def complete_the_sentence(masked_sentence):
    """
    Using the pre-trained model `albert-base-v2` to fill in the missing words in sentences.
    
    Args:
        masked_sentence (str): The sentence containing one or more [MASK] tokens to be predicted.

    Returns:
        list: A list of dictionaries containing predictions and scores, each dictionary representing a possible fill option.
    """
    unmasker = pipeline('fill-mask', model='albert-base-v2')
    return unmasker(masked_sentence)

# test_function_code --------------------

def test_complete_the_sentence():
    print("Testing started.")
    sentences_to_test = [
        "Tell me more about your [MASK] hobbies.",
        "I love going to the [MASK] on weekends.",
        "My favorite type of music is [MASK]."
    ]
    expected_num_results = 5

    for i, sentence in enumerate(sentences_to_test):
        print(f"Testing case [{i+1}/{len(sentences_to_test)}] started.")
        result = complete_the_sentence(sentence)
        assert isinstance(result, list), f"Test case [{i+1}/{len(sentences_to_test)}] failed: Result is not a list."
        assert len(result) == expected_num_results, f"Test case [{i+1}/{len(sentences_to_test)}] failed: Result does not contain {expected_num_results} items."
        print(f"Testing case [{i+1}/{len(sentences_to_test)}] finished.")
    print("Testing finished.")

test_complete_the_sentence()