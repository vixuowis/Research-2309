# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_named_entities(text):
    """
    Extracts named entities such as person names, locations, and organizations from a given text.

    Args:
        text (str): The text from which to extract named entities.

    Returns:
        list: A list of named entities found in the text. Each entity is represented as a dictionary with 'text' and 'type' keys.
    """
    # Load the NER model
    tagger = SequenceTagger.load('flair/ner-english')

    # Pass your text as a Sentence object
    sentence = Sentence(text)

    # Predict NER tags
    tagger.predict(sentence)

    # Extract named entities found in the text
    entities = []
    for entity in sentence.get_spans('ner'):
        entities.append({'text': entity.text, 'type': entity.tag})

    return entities

# test_function_code --------------------

def test_extract_named_entities():
    """
    Tests the extract_named_entities function.
    """
    # Test with a sentence containing person, location and organization names
    text = 'Barack Obama visited the White House yesterday.'
    entities = extract_named_entities(text)
    assert len(entities) == 3
    assert {'text': 'Barack Obama', 'type': 'PER'} in entities
    assert {'text': 'White House', 'type': 'LOC'} in entities

    # Test with a sentence containing no named entities
    text = 'The cat sat on the mat.'
    entities = extract_named_entities(text)
    assert len(entities) == 0

    # Test with a sentence containing multiple instances of the same named entity
    text = 'New York is a city in New York.'
    entities = extract_named_entities(text)
    assert len(entities) == 2
    assert {'text': 'New York', 'type': 'LOC'} in entities

    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_named_entities()