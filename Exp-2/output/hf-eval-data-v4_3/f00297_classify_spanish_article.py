# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_spanish_article(text, candidate_labels):
    """
    Classify a Spanish article into predefined sections.

    Args:
        text (str): The content of the article to be classified.
        candidate_labels (list of str): A list of sections to classify the article into.

    Returns:
        dict: The classification predictions with probabilities.

    Raises:
        ValueError: If `text` is not provided.
        ValueError: If `candidate_labels` is not provided or empty.
    """
    if not text:
        raise ValueError("No article text provided.")
    if not candidate_labels:
        raise ValueError("Candidate labels are not provided or the list is empty.")

    hypothesis_template = "Este ejemplo es {}."
    classifier = pipeline('zero-shot-classification', model='Recognai/bert-base-spanish-wwm-cased-xnli')
    predictions = classifier(text, candidate_labels, hypothesis_template=hypothesis_template)
    return predictions

# test_function_code --------------------

def test_classify_spanish_article():
    print("Testing started.")

    # Test case 1: Valid inputs
    print("Testing case [1/3] started.")
    article = "El autor se perfila, a los 50 a\u00f1os de su muerte, como uno de los grandes de su siglo."
    candidate_labels = ['cultura', 'sociedad', 'economia', 'salud', 'deportes']
    result = classify_spanish_article(article, candidate_labels)
    assert 'labels' in result, "Test case [1/3] failed: 'labels' key is not in the result."

    # Test case 2: Empty text
    print("Testing case [2/3] started.")
    try:
        result = classify_spanish_article('', candidate_labels)
    except ValueError as e:
        assert str(e) == "No article text provided.", "Test case [2/3] failed: Incorrect error message for empty text."

    # Test case 3: Empty candidate_labels
    print("Testing case [3/3] started.")
    try:
        result = classify_spanish_article(article, [])
    except ValueError as e:
        assert str(e) == "Candidate labels are not provided or the list is empty.", "Test case [3/3] failed: Incorrect error message for empty candidate labels."

    print("Testing finished.")



# call_test_function_line --------------------

test_classify_spanish_article()