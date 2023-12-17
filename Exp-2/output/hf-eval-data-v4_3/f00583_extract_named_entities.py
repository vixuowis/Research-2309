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
    Extracts names of people and companies mentioned in the text using a pre-trained NER model.

    Args:
        text (str): The input text from which entities are to be extracted.

    Returns:
        list: A list of entity names categorized as people or organizations.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('The input text must be a string.')
    ner_model = pipeline('ner', model='Jean-Baptiste/roberta-large-ner-english')
    ner_results = ner_model(text)
    entities = [result['word'] for result in ner_results if result['entity'] in ['PER', 'ORG']]
    return entities

# test_function_code --------------------

def test_extract_named_entities():
    print("Testing started.")
    sample_text = "Apple was founded by Steve Jobs."

    # Test case 1: Check if function returns a list.
    print("Testing case [1/3] started.")
    entities = extract_named_entities(sample_text)
    assert isinstance(entities, list), f"Test case [1/3] failed: Expected a list, got {type(entities)}"

    # Test case 2: Check if 'Apple' and 'Steve Jobs' are extracted.
    print("Testing case [2/3] started.")
    expected_entities = ['Apple', 'Steve Jobs']
    assert all(entity in entities for entity in expected_entities), f"Test case [2/3] failed: Expected entities not found in {entities}"

    # Test case 3: Check if ValueError is raised for non-string input.
    print("Testing case [3/3] started.")
    try:
        extract_named_entities(123)
        assert False, "Test case [3/3] failed: ValueError was not raised for non-string input."
    except ValueError:
        pass
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_named_entities()