# requirements_file --------------------

!pip install -U flair

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def detect_entities(text):
    """
    Detect named entities in a given text using an NER model.
    :param text: str - The input text to analyze.
    :return: List of named entities detected in the text.
    """
    # Load NER tagger model
    tagger = SequenceTagger.load('flair/ner-english-ontonotes-large')
    # Create a Sentence object
    sentence = Sentence(text)
    # Predict NER tags
    tagger.predict(sentence)
    # Extract entities
    entities = [entity for entity in sentence.get_spans('ner')]
    return entities

# test_function_code --------------------

def test_detect_entities():
    print("Testing started.")

    # Test case 1: Check if function detects personal names correctly
    print("Testing case [1/3] started.")
    text1 = "Elon Musk is working on SpaceX and Tesla." 
    entities1 = detect_entities(text1)
    assert any(entity.tag == 'PERSON' and 'Elon Musk' in entity.text for entity in entities1), "Test case [1/3] failed: 'Elon Musk' as a PERSON not detected."

    # Test case 2: Check if function detects locations correctly
    print("Testing case [2/3] started.")
    text2 = "The Eiffel Tower is located in Paris, France."
    entities2 = detect_entities(text2)
    assert any(entity.tag == 'LOC' and 'Paris' in entity.text for entity in entities2), "Test case [2/3] failed: 'Paris' as a LOC not detected."

    # Test case 3: Check for accurate detection of multiple entity types
    print("Testing case [3/3] started.")
    text3 = "Apple was founded by Steve Jobs in April, 1976."
    entities3 = detect_entities(text3)
    assert len(entities3) > 1, "Test case [3/3] failed: Multiple entities not detected."

    print("Testing finished.")

test_detect_entities()