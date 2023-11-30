# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def identify_entities(text):
    """
    Identify the names of people and locations mentioned in a text using Named Entity Recognition (NER).

    Args:
        text (str): The text in which to identify entities.

    Returns:
        list: A list of tuples where each tuple represents an entity. The first element of the tuple is the entity text and the second element is the entity type ('PER' for person, 'LOC' for location).
    """

    # Load model from disk
    tagger = SequenceTagger.load('ner')
    
    # Make sentence by sentence prediction with Flair
    sentence = Sentence(text)
    tagger.predict(sentence)

    # Extract the entities in a list of tuples
    entities = [(entity.text, entity.labels[0].value) for entity in sentence.get_spans('ner')] 
    
    return entities

# test_function_code --------------------

def test_identify_entities():
    assert identify_entities('George Washington went to Washington') == [('George Washington', 'PER'), ('Washington', 'LOC')]
    assert identify_entities('I live in New York and my friend John Doe lives in Los Angeles.') == [('New York', 'LOC'), ('John Doe', 'PER'), ('Los Angeles', 'LOC')]
    assert identify_entities('') == []
    return 'All Tests Passed'


# call_test_function_code --------------------

test_identify_entities()