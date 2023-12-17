# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_entities(text: str) -> dict:
    """Extract named entities from the text using BERT NER model.

    Args:
        text (str): The input text to process for named entity recognition.

    Returns:
        dict: A dictionary with entities and their respective types.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')

    nlp = pipeline('ner', model='dslim/bert-large-NER')
    entities = nlp(text)
    return entities

# test_function_code --------------------

def test_extract_entities():
    test_text = 'I recently purchased a MacBook Pro from Apple Inc. and had a fantastic customer support experience. John from their tech support team was incredibly helpful and professional.'
    expected_entities = [
        {'entity': 'B-PER', 'score': 0.995, 'index': 22, 'word': 'John'},
        {'entity': 'B-ORG', 'score': 0.998, 'index': 7, 'word': 'Apple'}
        # Note: The score and index values are illustrative and not the actual expected output from the model
    ]

    print('Testing started.')
    entities = extract_entities(test_text)
    assert any(ent['word'] == 'John' for ent in entities), 'Test failed: John should be identified as a person.'
    assert any(ent['word'] == 'Apple' for ent in entities), 'Test failed: Apple should be identified as an organization.'
    print('Testing finished.')

# call_test_function_line --------------------

test_extract_entities()