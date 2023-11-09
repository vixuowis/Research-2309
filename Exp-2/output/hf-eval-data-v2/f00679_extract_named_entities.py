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
        List of tuples, where each tuple represents a named entity and contains the entity value and its type.
    """
    # Load the NER model
    tagger = SequenceTagger.load('flair/ner-english')

    # Pass your text as a Sentence object
    sentence = Sentence(text)

    # Predict NER tags
    tagger.predict(sentence)

    # Extract named entities found in the text
    named_entities = [(entity.text, entity.tag) for entity in sentence.get_spans('ner')]

    return named_entities

# test_function_code --------------------

def test_extract_named_entities():
    """
    Tests the function extract_named_entities.
    """
    # Test with a sample sentence
    text = 'Barack Obama visited the White House yesterday.'
    named_entities = extract_named_entities(text)
    assert ('Barack Obama', 'PER') in named_entities
    assert ('White House', 'LOC') in named_entities

    # Test with another sample sentence
    text = 'Apple Inc. is planning to open a new store in San Francisco.'
    named_entities = extract_named_entities(text)
    assert ('Apple Inc.', 'ORG') in named_entities
    assert ('San Francisco', 'LOC') in named_entities

# call_test_function_code --------------------

test_extract_named_entities()