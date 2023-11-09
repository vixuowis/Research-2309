# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_named_entities(news_article):
    """
    Extract named entities from a news article using a pretrained BERT model.

    Args:
        news_article (str): The news article from which to extract named entities.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing the entity and its type.
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
    assert all(isinstance(entity, dict) for entity in entities)
    assert 'Los Angeles' in [entity['word'] for entity in entities]

# call_test_function_code --------------------

test_extract_named_entities()