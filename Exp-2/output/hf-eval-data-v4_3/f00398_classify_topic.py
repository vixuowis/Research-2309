# requirements_file --------------------

import subprocess

requirements = ["transformers", "sentence_transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_topic(sentence: str) -> str:
    """
    Classifies the topic of a given sentence into predefined categories.

    Args:
        sentence (str): The sentence to be classified.

    Returns:
        str: The category that the sentence most likely belongs to.

    Raises:
        ValueError: If the sentence is empty.
    """
    if not sentence:
        raise ValueError('The sentence provided is empty.')
    classifier = pipeline('zero-shot-classification', model='cross-encoder/nli-deberta-v3-xsmall')
    candidate_labels = ['technology', 'literature', 'science']
    result = classifier(sentence, candidate_labels)
    return result['labels'][0]

# test_function_code --------------------

def test_classify_topic():
    print("Testing started.")

    # Test case 1: Check for technology related sentence
    print("Testing case [1/3] started.")
    technology_sentence = 'Apple just announced the newest iPhone X'
    category = classify_topic(technology_sentence)
    assert category == 'technology', f"Test case [1/3] failed: Expected 'technology', got {category}"

    # Test case 2: Check for literature related sentence
    print("Testing case [2/3] started.")
    literature_sentence = 'Shakespeare wrote Romeo and Juliet during the Renaissance.'
    category = classify_topic(literature_sentence)
    assert category == 'literature', f"Test case [2/3] failed: Expected 'literature', got {category}"

    # Test case 3: Check for science related sentence
    print("Testing case [3/3] started.")
    science_sentence = 'Photosynthesis is essential for plant growth.'
    category = classify_topic(science_sentence)
    assert category == 'science', f"Test case [3/3] failed: Expected 'science', got {category}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_topic()