# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_german_text(text, categories):
    """
    Classify a given German text into predefined categories using a zero-shot classification model.

    Args:
        text (str): The German text to be classified.
        categories (list of str): A list of category names in German to classify the text into.

    Returns:
        dict: A dictionary with the input text, the predicted category, and confidence scores.

    Raises:
        ValueError: If `text` is empty or `categories` list is empty.

    """
    if not text:
        raise ValueError('Input text is empty.')
    if not categories:
        raise ValueError('Categories list is empty.')
    classifier = pipeline('zero-shot-classification', model='Sahajtomar/German_Zeroshot')
    hypothesis_template = 'In deisem geht es um {}.'
    results = classifier(text, categories, hypothesis_template=hypothesis_template)
    return results

# test_function_code --------------------

def test_classify_german_text():
    print("Testing started.")
    # Case 1: A text about tragedy
    print("Testing case [1/3] started.")
    result = classify_german_text('Letzte Woche gab es einen Selbstmord in einer nahe gelegenen kolonie', ['Verbrechen', 'Tragödie', 'Stehlen'])
    assert 'Tragödie' in result['labels'], "Test case [1/3] failed: Expected category 'Tragödie' not predicted."

    # Case 2: A text about crime
    print("Testing case [2/3] started.")
    result = classify_german_text('In der Stadt wurde eine Bank überfallen', ['Verbrechen', 'Tragödie', 'Stehlen'])
    assert 'Verbrechen' in result['labels'], "Test case [2/3] failed: Expected category 'Verbrechen' not predicted."

    # Case 3: A text about theft
    print("Testing case [3/3] started.")
    result = classify_german_text('Jemand hat mein Fahrrad gestohlen', ['Verbrechen', 'Tragödie', 'Stehlen'])
    assert 'Stehlen' in result['labels'], "Test case [3/3] failed: Expected category 'Stehlen' not predicted."
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_german_text()