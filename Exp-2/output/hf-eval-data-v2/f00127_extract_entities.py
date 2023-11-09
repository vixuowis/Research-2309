# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_entities(news_article):
    """
    Extracts named entities such as people, organizations, and locations from a news article.

    Args:
        news_article (str): The news article from which to extract entities.

    Returns:
        list: A list of dictionaries. Each dictionary represents an entity and contains the entity string, its start and end indices in the input string, and its type (e.g., 'PER' for person, 'ORG' for organization, 'LOC' for location).
    """
    tokenizer = AutoTokenizer.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    model = AutoModelForTokenClassification.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    ner_results = nlp(news_article)
    return ner_results

# test_function_code --------------------

def test_extract_entities():
    """
    Tests the extract_entities function.
    """
    news_article = 'Nader Jokhadar had given Syria the lead with a well-struck header in the seventh minute.'
    entities = extract_entities(news_article)
    assert isinstance(entities, list)
    assert len(entities) > 0
    for entity in entities:
        assert 'entity' in entity
        assert 'start' in entity
        assert 'end' in entity
        assert 'index' in entity
        assert 'score' in entity
        assert 'is_subword' in entity

# call_test_function_code --------------------

test_extract_entities()