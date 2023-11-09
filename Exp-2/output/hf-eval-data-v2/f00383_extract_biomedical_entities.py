# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_biomedical_entities(case_report_text):
    """
    Extract biomedical entities from a given set of case reports.

    Args:
        case_report_text (str): The text of the case report from which to extract biomedical entities.

    Returns:
        List[Dict]: A list of dictionaries, each containing information about a detected entity.
    """
    ner_pipeline = pipeline('ner', model='d4data/biomedical-ner-all', tokenizer='d4data/biomedical-ner-all', aggregation_strategy='simple')
    entities = ner_pipeline(case_report_text)
    return entities

# test_function_code --------------------

def test_extract_biomedical_entities():
    """
    Test the function extract_biomedical_entities.
    """
    case_report_text = 'The patient reported no recurrence of palpitations at follow-up 6 months after the ablation.'
    entities = extract_biomedical_entities(case_report_text)
    assert isinstance(entities, list), 'The result should be a list.'
    assert len(entities) > 0, 'The list should not be empty.'
    for entity in entities:
        assert 'entity' in entity, 'Each entity should have an entity field.'
        assert 'score' in entity, 'Each entity should have a score field.'
        assert 'index' in entity, 'Each entity should have an index field.'
        assert 'start' in entity, 'Each entity should have a start field.'
        assert 'end' in entity, 'Each entity should have an end field.'

# call_test_function_code --------------------

test_extract_biomedical_entities()