# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_entities(text):
    """
    Extracts the names of companies and people mentioned in the text using Named Entity Recognition.

    Args:
        text (str): The text from which to extract entities.

    Returns:
        entities (list): A list of entities (people and companies) extracted from the text.
    """
    ner_model = pipeline('ner', model='Jean-Baptiste/roberta-large-ner-english')
    ner_results = ner_model(text)
    entities = [result['word'] for result in ner_results if result['entity'] in ['PER', 'ORG']]
    return entities

# test_function_code --------------------

def test_extract_entities():
    """
    Tests the function extract_entities.
    """
    text = "Apple was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne to develop and sell Wozniak's Apple I personal computer."
    entities = extract_entities(text)
    assert 'Apple' in entities
    assert 'Steve Jobs' in entities
    assert 'Steve Wozniak' in entities
    assert 'Ronald Wayne' in entities

# call_test_function_code --------------------

test_extract_entities()