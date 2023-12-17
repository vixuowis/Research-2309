# requirements_file --------------------

!pip install -U transformers==4.18.0

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_biomedical_entities(text):
    """
    Extract biomedical entities from the given text using a named entity recognition (NER) model.

    Parameters:
    text (str): The case report text from which to extract entities.

    Returns:
    list: A list of entities extracted from the text.
    """
    # Initialize the NER pipeline
    ner_pipeline = pipeline('ner', model='d4data/biomedical-ner-all', tokenizer='d4data/biomedical-ner-all', aggregation_strategy='simple')
    
    # Extract entities
    entities = ner_pipeline(text)
    return entities

# test_function_code --------------------

def test_extract_biomedical_entities():
    print("Testing extract_biomedical_entities function.")
    sample_text = 'The patient reported no recurrence of palpitations at follow-up 6 months after the ablation.'

    # Test case 1: Check if the function returns a list
    print("Testing case [1/2] started.")
    entities = extract_biomedical_entities(sample_text)
    assert isinstance(entities, list), "Test case [1/2] failed: The function should return a list of entities."

    # Test case 2: Test with a known sample to check for specific entities
    print("Testing case [2/2] started.")
    known_entities = ['palpitations', 'ablation']
    for entity in known_entities:
        assert any(entity in e['word'] for e in entities), f"Test case [2/2] failed: Expected entity '{entity}' not found in the output."
    print("Testing completed successfully.")