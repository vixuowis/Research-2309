# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_entities(text):
    """
    Extracts names of people, organizations, and locations from a given text.

    Args:
        text (str): The text from which to extract entities.

    Returns:
        list: A list of dictionaries. Each dictionary represents an entity and contains the entity's word, score, entity type, and index.
    """
    tokenizer = AutoTokenizer.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    model = AutoModelForTokenClassification.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    return nlp(text)

# test_function_code --------------------

def test_extract_entities():
    """
    Tests the extract_entities function.
    """
    example = 'John Doe works at Google headquarters in Mountain View, California.'
    result = extract_entities(example)
    assert isinstance(result, list)
    assert 'John Doe' in [entity['word'] for entity in result]
    assert 'Google' in [entity['word'] for entity in result]
    assert 'Mountain View' in [entity['word'] for entity in result]

# call_test_function_code --------------------

test_extract_entities()