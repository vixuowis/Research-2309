# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_named_entities(text):
    """
    Extract named entities from a given text using the 'flair/ner-english-ontonotes-fast' model.

    Args:
        text (str): The text from which to extract named entities.

    Returns:
        list: A list of named entities extracted from the text.
    """
    tagger = SequenceTagger.load('flair/ner-english-ontonotes-fast')
    sentence = Sentence(text)
    tagger.predict(sentence)
    named_entities = [entity for entity in sentence.get_spans('ner')]
    return named_entities

# test_function_code --------------------

def test_extract_named_entities():
    assert len(extract_named_entities('On September 1st George Washington won 1 dollar.')) == 2
    assert len(extract_named_entities('Apple Inc. is planning to open a new store in San Francisco.')) == 2
    assert len(extract_named_entities('')) == 0
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_named_entities()