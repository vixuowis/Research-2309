# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_named_entities(text):
    """
    Extract named entities from a given text using the 'flair/ner-english-ontonotes-fast' NER model.

    Args:
        text (str): The input text from which to extract named entities.

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
    """
    Test the extract_named_entities function.
    """
    text = 'On September 1st George Washington won 1 dollar.'
    named_entities = extract_named_entities(text)
    assert len(named_entities) > 0, 'No named entities found.'
    assert any(entity.tag == 'PERSON' for entity in named_entities), 'No person named entities found.'

# call_test_function_code --------------------

test_extract_named_entities()