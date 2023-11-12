# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_entities(text):
    """
    Extract names of people, organizations, and locations from a given text.

    Args:
        text (str): The text from which to extract entities.

    Returns:
        list: A list of dictionaries. Each dictionary contains the entity, its start and end indices in the text, and its type (person, organization, or location).
    """
    tokenizer = AutoTokenizer.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    model = AutoModelForTokenClassification.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    return nlp(text)

# test_function_code --------------------

def test_extract_entities():
    """
    Test the extract_entities function.
    """
    # Test with English text
    result = extract_entities('John Doe works at Google headquarters in Mountain View, California.')
    assert 'John Doe' in [entity['word'] for entity in result]
    assert 'Google' in [entity['word'] for entity in result]
    assert 'Mountain View' in [entity['word'] for entity in result]
    assert 'California' in [entity['word'] for entity in result]

    # Test with German text
    result = extract_entities('John Doe arbeitet im Google-Hauptquartier in Mountain View, Kalifornien.')
    assert 'John Doe' in [entity['word'] for entity in result]
    assert 'Google' in [entity['word'] for entity in result]
    assert 'Mountain View' in [entity['word'] for entity in result]
    assert 'Kalifornien' in [entity['word'] for entity in result]

    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_entities()