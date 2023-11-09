from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def extract_entities(news_article):
    """
    This function extracts named entities such as people, organizations, and locations from a given news article.
    It uses the 'Davlan/distilbert-base-multilingual-cased-ner-hrl' model from the Transformers library.
    
    Parameters:
    news_article (str): The news article from which to extract entities.
    
    Returns:
    list: A list of dictionaries, each containing information about a named entity.
    """
    tokenizer = AutoTokenizer.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    model = AutoModelForTokenClassification.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    ner_results = nlp(news_article)
    return ner_results