# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_named_entities(news_article):
    """
    Extract named entities from a news article using a pretrained BERT model.

    Args:
        news_article (str): The news article from which to extract named entities.

    Returns:
        list: A list of dictionaries. Each dictionary represents a named entity and contains the entity, its label, and its index in the text.

    Raises:
        OSError: If there is a problem loading the model or processing the text.
    """
    ner_model = pipeline('ner', model='dslim/bert-base-NER-uncased')
    entities = ner_model(news_article)
    return entities

# test_function_code --------------------

def test_extract_named_entities():
    """
    Test the extract_named_entities function.
    """
    news_article = "Large parts of Los Angeles have been hit by power outages with electricity provider Southern California Edison pointing at high winds as the cause for the disruption. Thousands of residents..."
    entities = extract_named_entities(news_article)
    assert isinstance(entities, list)
    assert len(entities) > 0
    assert 'entity' in entities[0]
    assert 'label' in entities[0]
    assert 'index' in entities[0]
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_named_entities()