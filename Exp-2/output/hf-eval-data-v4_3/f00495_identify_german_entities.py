# requirements_file --------------------

import subprocess

requirements = ["flair"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def identify_german_entities(text: str) -> str:
    """
    Identify named entities in a given German text.

    Args:
        text (str): The text in German to analyze.

    Returns:
        str: A string representation of the named entities found.

    Raises:
        ValueError: If the text is empty or None.
    """
    if not text:
        raise ValueError('Input text cannot be empty')

    tagger = SequenceTagger.load('flair/ner-german')
    sentence = Sentence(text)
    tagger.predict(sentence)
    return '\n'.join(str(entity) for entity in sentence.get_spans('ner'))

# test_function_code --------------------

def test_identify_german_entities():
    print("Testing started.")
    sample_text = 'George Washington ging nach Washington.'

    # Test case 1: Check if function returns a string
    print("Testing case [1/3] started.")
    result = identify_german_entities(sample_text)
    assert isinstance(result, str), f"Test case [1/3] failed: Result is not a string."

    # Test case 2: Check for non-empty text
    print("Testing case [2/3] started.")
    try:
        identify_german_entities('')
        assert False, "Test case [2/3] failed: ValueError not raised for empty string."
    except ValueError:
        assert True

    # Test case 3: Check for the detection of named entities
    print("Testing case [3/3] started.")
    result = identify_german_entities(sample_text)
    assert 'Washington' in result, f"Test case [3/3] failed: Expected named entity not found in result."
    print("Testing finished.")

# call_test_function_line --------------------

test_identify_german_entities()