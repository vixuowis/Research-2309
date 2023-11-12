# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_topic(sentence: str) -> str:
    """
    Classify the topic of a given sentence among 'technology', 'literature', and 'science'.

    Args:
        sentence (str): The sentence to be classified.

    Returns:
        str: The classified topic of the sentence.

    Raises:
        ValueError: If the input sentence is not a string.
    """
    if not isinstance(sentence, str):
        raise ValueError('Input sentence must be a string.')

    classifier = pipeline('zero-shot-classification', model='cross-encoder/nli-deberta-v3-xsmall')
    candidate_labels = ['technology', 'literature', 'science']
    result = classifier(sentence, candidate_labels)
    return result['labels'][0]

# test_function_code --------------------

def test_classify_topic():
    assert classify_topic('Apple just announced the newest iPhone X') == 'technology'
    assert classify_topic('The Great Gatsby is a novel written by F. Scott Fitzgerald') == 'literature'
    assert classify_topic('Einstein developed the theory of relativity') == 'science'
    assert classify_topic('') == 'ValueError'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_topic()