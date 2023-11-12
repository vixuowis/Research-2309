# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_entities(text):
    '''
    Extracts named entities from a given text using the 'dslim/bert-large-NER' model from the transformers library.
    
    Args:
        text (str): The text from which to extract named entities.
    
    Returns:
        A list of dictionaries. Each dictionary represents a named entity and contains the entity, its start and end indices in the text, and its type (e.g., 'PER' for person, 'ORG' for organization).
    '''
    nlp = pipeline('ner', model='dslim/bert-large-NER')
    entities = nlp(text)
    return entities

# test_function_code --------------------

def test_extract_entities():
    '''
    Tests the extract_entities function with various test cases.
    '''
    # Test case 1: Text with person and organization entities
    text1 = 'I recently purchased a MacBook Pro from Apple Inc. and had a fantastic customer support experience. John from their tech support team was incredibly helpful and professional.'
    entities1 = extract_entities(text1)
    assert len(entities1) > 0 and any(entity['entity'] == 'John' for entity in entities1)
    
    # Test case 2: Text with no entities
    text2 = 'I recently purchased a laptop and had a fantastic customer support experience. The tech support team was incredibly helpful and professional.'
    entities2 = extract_entities(text2)
    assert len(entities2) == 0
    
    # Test case 3: Text with multiple entities of the same type
    text3 = 'Bill and Melinda Gates founded the Bill & Melinda Gates Foundation.'
    entities3 = extract_entities(text3)
    assert len(entities3) > 0 and all(entity['entity'] in ['Bill', 'Melinda', 'Gates', 'Bill & Melinda Gates Foundation'] for entity in entities3)
    
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_entities()