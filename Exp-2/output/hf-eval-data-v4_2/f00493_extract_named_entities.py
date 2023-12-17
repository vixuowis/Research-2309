# requirements_file --------------------

!pip install -U flair

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_named_entities(text):
    """
    Extracts named entities from a given text snippet using Flair's SequenceTagger.

    Args:
        text (str): The text snippet from which to extract named entities.

    Returns:
        list: A list of named entity spans found in the text.

    Raises:
        ValueError: If an empty string is provided as input.

    """
    if not text:
        raise ValueError('Input text must not be empty')
    # Load the pre-trained NER model
    tagger = SequenceTagger.load('flair/ner-english-ontonotes')
    # Create a Sentence object from the input text
    sentence = Sentence(text)
    # Predict named entities using the model
    tagger.predict(sentence)
    # Extract and return the named entities
    return sentence.get_spans('ner')

# test_function_code --------------------

def test_extract_named_entities():
    print('Testing started.')
    # Test case 1: Valid text with known entities
    print('Testing case [1/3] started.')
    entities = extract_named_entities('Jane Smith visited the Empire State Building on June 7th.')
    expected_entities = ['Jane Smith', 'Empire State Building', 'June 7th']
    assert all(str(entity) in expected_entities for entity in entities), f'Test case [1/3] failed: unexpected entities {entities}'

    # Test case 2: Empty string as input
    print('Testing case [2/3] started.')
    try:
        extract_named_entities('')
        assert False, 'Test case [2/3] failed: ValueError not raised for empty input'
    except ValueError as e:
        assert str(e) == 'Input text must not be empty', f'Test case [2/3] failed: {e}'

    # Test case 3: Text with no entities
    print('Testing case [3/3] started.')
    entities = extract_named_entities('This is a generic sentence without named entities.')
    assert not entities, f'Test case [3/3] failed: expected no entities but got {entities}'
    print('Testing finished.')

# call_test_function_line --------------------

test_extract_named_entities()