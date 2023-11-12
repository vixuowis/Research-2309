# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_entities(news_article_text):
    '''
    Extract entities from a given news article text using the pre-trained model 'flair/ner-english-ontonotes'.

    Args:
        news_article_text (str): The text of the news article.

    Returns:
        List of entities extracted from the news article text.
    '''
    tagger = SequenceTagger.load('flair/ner-english-ontonotes')
    sentence = Sentence(news_article_text)
    tagger.predict(sentence)
    entities = sentence.get_spans('ner')
    return entities

# test_function_code --------------------

def test_extract_entities():
    '''
    Test the function extract_entities.
    '''
    test_text_1 = 'On September 1st George Washington won 1 dollar.'
    test_text_2 = 'Apple Inc. is planning to open a new store in San Francisco.'
    test_text_3 = 'The United Nations will hold a meeting on climate change in Paris.'
    assert len(extract_entities(test_text_1)) > 0
    assert len(extract_entities(test_text_2)) > 0
    assert len(extract_entities(test_text_3)) > 0
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_entities()