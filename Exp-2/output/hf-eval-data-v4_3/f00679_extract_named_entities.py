# requirements_file --------------------

import subprocess

requirements = ["flair"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from flair.models import SequenceTagger
from flair.data import Sentence

# function_code --------------------

def extract_named_entities(text):
    """
    Extract named entities from a given text using Flair NER model.

    Args:
        text (str): The input text from which to extract named entities.

    Returns:
        list: A list of named entities found in the text.

    Raises:
        ValueError: If the input text is empty.
    """
    if not text:
        raise ValueError('Input text cannot be empty.')

    # Load the Flair NER model
    tagger = SequenceTagger.load('flair/ner-english')

    # Create a Sentence object from the input text
    sentence = Sentence(text)

    # Apply the NER tagger to the sentence
    tagger.predict(sentence)

    # Get the list of named entities
    named_entities = sentence.get_spans('ner')

    # Extract the text and tag of each named entity
    entity_list = [(entity.text, entity.tags['ner'].value) for entity in named_entities]

    return entity_list

# test_function_code --------------------

def test_extract_named_entities():
    print("Testing started.")

    # Test case 1: Check empty text
    print("Testing case [1/2] started.")
    try:
        extract_named_entities('')
        assert False, "Test case [1/2] failed: Empty text did not raise ValueError."
    except ValueError:
        pass

    # Test case 2: Check normal text
    print("Testing case [2/2] started.")
    entities = extract_named_entities('George Washington went to Washington.')
    expected_entities = [('George Washington', 'PER'), ('Washington', 'LOC')]
    assert entities == expected_entities, f"Test case [2/2] failed: Expected {expected_entities}, got {entities}."
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_named_entities()