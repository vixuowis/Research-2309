from flair.data import Sentence
from flair.models import SequenceTagger


def extract_named_entities(news_article):
    """
    This function extracts all the well-known named entities such as person names, locations, and organizations from a given news article.
    
    Parameters:
    news_article (str): The news article from which to extract named entities.
    
    Returns:
    List[str]: A list of named entities found in the news article.
    """
    # Load the NER model
    tagger = SequenceTagger.load('flair/ner-english')

    # Pass your news article as a Sentence object
    sentence = Sentence(news_article)

    # Predict NER tags
    tagger.predict(sentence)

    # Extract named entities found in the text
    named_entities = [str(entity) for entity in sentence.get_spans('ner')]

    return named_entities