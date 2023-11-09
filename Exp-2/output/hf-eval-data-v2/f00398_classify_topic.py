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
    """
    classifier = pipeline('zero-shot-classification', model='cross-encoder/nli-deberta-v3-xsmall')
    candidate_labels = ['technology', 'literature', 'science']
    result = classifier(sentence, candidate_labels)
    return result['labels'][0]

# test_function_code --------------------

def test_classify_topic():
    """
    Test the classify_topic function.
    """
    sentence = 'Apple just announced the newest iPhone X'
    assert classify_topic(sentence) == 'technology'
    sentence = 'The universe is expanding at an accelerating rate'
    assert classify_topic(sentence) == 'science'
    sentence = 'Shakespeare was a great playwright'
    assert classify_topic(sentence) == 'literature'

# call_test_function_code --------------------

test_classify_topic()