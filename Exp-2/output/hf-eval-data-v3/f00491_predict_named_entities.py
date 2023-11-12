# function_import --------------------

from transformers import pipeline

# function_code --------------------

def predict_named_entities(article_text):
    """
    Predict the named entities from the given article text.

    Args:
        article_text (str): The text of the article.

    Returns:
        list: A list of dictionaries. Each dictionary represents a named entity and contains the entity, its start and end indices in the text, and its type.
    """
    nlp = pipeline('ner', model='dslim/bert-base-NER-uncased')
    entities = nlp(article_text)
    return entities

# test_function_code --------------------

def test_predict_named_entities():
    """
    Test the predict_named_entities function.
    """
    article_text = 'My name is John and I live in New York.'
    entities = predict_named_entities(article_text)
    assert isinstance(entities, list)
    assert len(entities) > 0
    assert 'entity' in entities[0]
    assert 'start' in entities[0]
    assert 'end' in entities[0]
    assert 'index' in entities[0]
    assert 'score' in entities[0]
    assert 'is_subword' in entities[0]
    assert 'start' in entities[0]
    assert 'end' in entities[0]
    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_named_entities()