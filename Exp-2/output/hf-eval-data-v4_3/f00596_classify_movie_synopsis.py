# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_movie_synopsis(synopsis: str) -> dict:
    """
    Classify a movie synopsis into crime, tragedy, or theft categories.

    Args:
        synopsis (str): The movie synopsis in German.

    Returns:
        dict: A dictionary containing the classification label and score.

    """
    classifier = pipeline('zero-shot-classification', model='Sahajtomar/German_Zeroshot')
    labels = ['Verbrechen', 'Tragödie', 'Stehlen']
    hypothesis_template = 'In diesem Film geht es um {}'
    result = classifier(synopsis, labels, hypothesis_template=hypothesis_template)
    return result

# test_function_code --------------------

def test_classify_movie_synopsis():
    print("Testing started.")
    # Use a sample synopsis for testing
    sample_synopsis = 'Letzte Woche gab es einen Mord in einer nahe gelegenen Kolonie.'

    # Test case 1
    print("Testing case [1/1] started.")
    result = classify_movie_synopsis(sample_synopsis)
    assert 'label' in result and result['label'] in ['Verbrechen', 'Tragödie', 'Stehlen'], f"Test case [1/1] failed: Unexpected result {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_movie_synopsis()