# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def predict_named_entities(text):
    """
    Predict the named entities in a given piece of text using a pre-trained NER model.

    Args:
        text (str): The text in which to predict named entities.

    Returns:
        list: A list of dicts containing the detected entities and their types.

    Raises:
        ValueError: If the text argument is not a string or is empty.
    """
    if not isinstance(text, str) or not text:
        raise ValueError('The text argument must be a non-empty string.')
    nlp = pipeline('ner', model='dslim/bert-base-NER-uncased')
    entities = nlp(text)
    return entities

# test_function_code --------------------

def test_predict_named_entities():
    print("Testing started.")
    sample_texts = [
        'My name is John and I live in New York.',
        '',
        123
    ]

    # Test case 1: Valid text
    print('Testing case [1/3] started.')
    result_1 = predict_named_entities(sample_texts[0])
    assert isinstance(result_1, list), 'Test case [1/3] failed: The result should be a list.'

    # Test case 2: Empty text
    print('Testing case [2/3] started.')
    try:
        predict_named_entities(sample_texts[1])
    except ValueError as e:
        assert str(e) == 'The text argument must be a non-empty string.', 'Test case [2/3] failed: ValueError not raised for empty string.'

    # Test case 3: Non-string text
    print('Testing case [3/3] started.')
    try:
        predict_named_entities(sample_texts[2])
    except ValueError as e:
        assert str(e) == 'The text argument must be a non-empty string.', 'Test case [3/3] failed: ValueError not raised for non-string input.'
    print('Testing finished.')

# call_test_function_line --------------------

test_predict_named_entities()