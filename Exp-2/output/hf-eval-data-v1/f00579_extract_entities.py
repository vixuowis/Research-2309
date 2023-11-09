from transformers import pipeline


def extract_entities(news_article):
    """
    This function uses the 'dslim/bert-base-NER-uncased' model from the transformers library to extract named entities from a given news article.
    Named entities can include person names, locations, and organizations.
    The function returns a list of entities extracted from the news article.
    """
    # Create a named entity recognition (NER) model using the pipeline function
    ner_model = pipeline('ner', model='dslim/bert-base-NER-uncased')
    
    # Use the NER model to process the news article and extract named entities
    entities = ner_model(news_article)
    
    return entities