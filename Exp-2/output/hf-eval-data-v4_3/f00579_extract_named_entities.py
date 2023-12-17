# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_named_entities(text):
    """
    Extract named entities from the input text using a pre-trained NER model.

    Args:
        text (str): The input text from which to extract named entities.

    Returns:
        list: A list of named entity objects each containing entity text and type.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')
    ner_model = pipeline('ner', model='dslim/bert-base-NER-uncased')
    return ner_model(text)

# test_function_code --------------------

def test_extract_named_entities():
    print("Testing started.")

    # Test case 1: Check if the function correctly extracts named entities from a sample text.
    print("Testing case [1/1] started.")
    sample_text = "Thousands of people in San Francisco have protested against government policies."
    result = extract_named_entities(sample_text)
    expected_entities = ['San Francisco', 'government policies']
    assert all(entity['word'] in result for entity in expected_entities), f"Test case [1/1] failed: Expected entities {expected_entities} not found in result {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_named_entities()