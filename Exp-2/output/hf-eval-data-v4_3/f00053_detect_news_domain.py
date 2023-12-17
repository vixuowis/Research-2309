# requirements_file --------------------

import subprocess

requirements = ["transformers", "sentence_transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_news_domain(text, model='cross-encoder/nli-roberta-base'):
    """
    Detects the domain of a news article using a zero-shot classification approach.

    Args:
        text (str): The news article text to classify.
        model (str): The name of the pre-trained model to use for classification.
                   Default is 'cross-encoder/nli-roberta-base'.

    Returns:
        dict: A dictionary containing the classification label and the associated
              probabilities of all candidate labels.

    Raises:
        ValueError: If the provided text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('The input text must be a string.')
    candidate_labels = ['technology', 'sports', 'politics']
    classifier = pipeline('zero-shot-classification', model=model)
    return classifier(text, candidate_labels)

# test_function_code --------------------

def test_detect_news_domain():
    print('Testing started.')
    # Test case: Classifying technology news text
    print('Testing case [1/3] started.')
    result = detect_news_domain('Latest trends in AI and machine learning')
    assert 'technology' in result['labels'], 'Test case [1/3] failed: Expected technology label.'
    # Test case: Classifying sports news text
    print('Testing case [2/3] started.')
    result = detect_news_domain('Champions League last night results')
    assert 'sports' in result['labels'], 'Test case [2/3] failed: Expected sports label.'
    # Test case: Classifying politics news text
    print('Testing case [3/3] started.')
    result = detect_news_domain('Election results show a shift in voter sentiment')
    assert 'politics' in result['labels'], 'Test case [3/3] failed: Expected politics label.'
    print('Testing finished.')

# call_test_function_line --------------------

test_detect_news_domain()