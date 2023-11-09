# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def identify_entities(text):
    """
    Identify the names of people and locations mentioned in a text using Named Entity Recognition (NER).

    Args:
        text (str): The text to analyze.

    Returns:
        list: A list of tuples where each tuple represents an entity. The first element of the tuple is the entity name and the second element is the entity type ('PER' for person, 'LOC' for location).
    """
    tagger = SequenceTagger.load('flair/ner-english')
    sentence = Sentence(text)
    tagger.predict(sentence)
    entities = []
    for entity in sentence.get_spans('ner'):
        if entity.tag == 'PER' or entity.tag == 'LOC':
            entities.append((entity.text, entity.tag))
    return entities

# test_function_code --------------------

def test_identify_entities():
    """
    Test the identify_entities function.
    """
    text = 'George Washington went to Washington'
    expected_result = [('George Washington', 'PER'), ('Washington', 'LOC')]
    assert identify_entities(text) == expected_result

# call_test_function_code --------------------

test_identify_entities()