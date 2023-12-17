# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_biomed_entities(text):
    """
    Extract biomedical entities from the given text using a pre-trained NER model.

    Args:
        text (str): A string containing the case report or text from which to extract entities.

    Returns:
        List[Dict[str, str]]: A list of dictionaries where each dict represents an entity with keys 'entity', 'score', 'index', and 'word'.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')

    ner_pipeline = pipeline('ner', model='d4data/biomedical-ner-all', tokenizer='d4data/biomedical-ner-all', aggregation_strategy='simple')

    return ner_pipeline(text)

# test_function_code --------------------

def test_extract_biomed_entities():
    print("Testing started.")
    sample_text = 'The patient reported no recurrence of palpitations at follow-up 6 months after the ablation.'

    # Testing case 1: Valid string input
    print("Testing case [1/1] started.")
    entities = extract_biomed_entities(sample_text)
    assert isinstance(entities, list) and all(isinstance(entity, dict) for entity in entities), "Test case [1/1] failed: The function should return a list of dictionaries."
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_biomed_entities()