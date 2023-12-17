# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_review(review_text, category_labels):
    """
    Classifies a given review text into one of the provided category labels using zero-shot classification.

    Args:
        review_text (str): The review text to classify.
        category_labels (list): A list of strings representing the category labels to classify the text into.

    Returns:
        dict: A dictionary containing the 'labels' and 'scores' predicted by the classifier.

    Raises:
        ValueError: If the `category_labels` list is empty.
    """
    if not category_labels:
        raise ValueError('Category labels list cannot be empty.')

    classifier = pipeline('zero-shot-classification', model='vicgalle/xlm-roberta-large-xnli-anli')
    result = classifier(review_text, category_labels)
    return result

# test_function_code --------------------

def test_classify_review():
    print("Testing started.")

    # Test case 1: The review text is in Spanish and classification for cooking, dancing, and travel.
    print("Testing case [1/3] started.")
    review_text = "La paella que comí en mi último viaje fue excepcional."
    category_labels = ['cocina', 'danza', 'viaje']
    result = classify_review(review_text, category_labels)
    assert 'labels' in result and 'scores' in result, "Test case [1/3] failed: Result should have 'labels' and 'scores'"

    # Test case 2: The review text is an empty string.
    print("Testing case [2/3] started.")
    review_text = ""
    result = classify_review(review_text, category_labels)
    assert 'labels' in result and 'scores' in result, "Test case [2/3] failed: Result should have 'labels' and 'scores'"

    # Test case 3: The category_labels list is empty.
    print("Testing case [3/3] started.")
    category_labels = []
    try:
        classify_review(review_text, category_labels)
        assert False, "Test case [3/3] failed: ValueError expected"
    except ValueError as e:
        assert str(e) == 'Category labels list cannot be empty.', "Test case [3/3] failed: Incorrect ValueError message"

    print("Testing finished.")

# call_test_function_line --------------------

test_classify_review()