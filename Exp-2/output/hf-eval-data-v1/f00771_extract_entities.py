from flair.data import Sentence
from flair.models import SequenceTagger

def extract_entities(news_article_text):
    """
    Extract entities from a given news article text using the pre-trained model 'flair/ner-english-ontonotes'.

    Args:
        news_article_text (str): The text of the news article.

    Returns:
        List of entities extracted from the news article text.
    """
    tagger = SequenceTagger.load('flair/ner-english-ontonotes')
    sentence = Sentence(news_article_text)
    tagger.predict(sentence)
    entities = sentence.get_spans('ner')
    return entities