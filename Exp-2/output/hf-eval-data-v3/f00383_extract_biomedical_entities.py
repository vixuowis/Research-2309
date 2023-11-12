# function_import --------------------

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# function_code --------------------

def extract_biomedical_entities(text):
    """
    Extract biomedical entities from a given text using a Named Entity Recognition (NER) model.

    Args:
        text (str): The text from which to extract biomedical entities.

    Returns:
        list: A list of dictionaries. Each dictionary represents an entity and contains the entity, its start and end indices in the text, and its score.
    """
    ner_pipeline = pipeline('ner', model='d4data/biomedical-ner-all', tokenizer='d4data/biomedical-ner-all', aggregation_strategy='simple')
    entities = ner_pipeline(text)
    return entities

# test_function_code --------------------

def test_extract_biomedical_entities():
    """
    Test the function extract_biomedical_entities.
    """
    # Test case: A sentence with biomedical entities
    text = 'The patient reported no recurrence of palpitations at follow-up 6 months after the ablation.'
    entities = extract_biomedical_entities(text)
    assert isinstance(entities, list), 'The result should be a list.'
    for entity in entities:
        assert 'entity' in entity, 'Each entity should have an entity field.'
        assert 'start' in entity, 'Each entity should have a start field.'
        assert 'end' in entity, 'Each entity should have an end field.'
        assert 'score' in entity, 'Each entity should have a score field.'

    # Test case: An empty string
    text = ''
    entities = extract_biomedical_entities(text)
    assert entities == [], 'The result should be an empty list for an empty string.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_biomedical_entities()