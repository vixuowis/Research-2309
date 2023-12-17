# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_headline(headline: str, candidate_labels: list) -> dict:
    """
    Classify news headlines into categories using a zero-shot classifier.

    Args:
        headline: A string representing the news headline to classify.
        candidate_labels: A list of strings representing the candidate categories.

    Returns:
        A dictionary containing the classification results.

    Raises:
        ValueError: If `headline` is not a string or `candidate_labels` is not a list.
    """
    if not isinstance(headline, str) or not isinstance(candidate_labels, list):
        raise ValueError('The headline must be a string and candidate_labels must be a list.')
    
    headlines_classifier = pipeline('zero-shot-classification', model='cross-encoder/nli-deberta-v3-xsmall')
    return headlines_classifier(headline, candidate_labels)

# test_function_code --------------------

def test_classify_headline():
    print("Testing started.")

    headline = "Apple just announced the newest iPhone X"
    candidate_labels = ['technology', 'sports', 'politics']

    # Test case 1: Check if the function outputs a dictionary
    print("Testing case [1/3] started.")
    result = classify_headline(headline, candidate_labels)
    assert isinstance(result, dict), f"Test case [1/3] failed: Expected a dictionary, got {type(result)}"

    # Test case 2: Check if the function fails with wrong types
    print("Testing case [2/3] started.")
    try:
        classify_headline(123, 'not a list')
        assert False, "Test case [2/3] failed: Expected ValueError for invalid arguments"
    except ValueError:
        pass

    # Test case 3: Check if the labels in the result are from the candidate labels
    print("Testing case [3/3] started.")
    for label in result.get('labels', []):
        assert label in candidate_labels, f"Test case [3/3] failed: Label {label} is not in candidate labels"
    
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_headline()