# requirements_file --------------------

!pip install -U flair

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_named_entities(text):
    """
    Extract named entities from a given text using the Flair NER model.

    :param text: str - The input text from which named entities are to be extracted.
    :return: list - A list of named entities extracted from the text.
    """
    # Load the NER model
    tagger = SequenceTagger.load('flair/ner-english-ontonotes-fast')

    # Prepare the input text
    sentence = Sentence(text)

    # Predict named entities
    tagger.predict(sentence)

    # Extract named entities
    named_entities = [entity for entity in sentence.get_spans('ner')]

    return named_entities

# test_function_code --------------------

def test_extract_named_entities():
    print("Testing extract_named_entities function.")
    # Example text
    text = "On September 1st, George Washington won 1 dollar."

    # Test case: Extracting named entities
    expected_entities = ['George Washington', 'September 1st', '1 dollar']
    actual_entities = extract_named_entities(text)
    assert all(str(entity) in expected_entities for entity in actual_entities), f"Test failed: {actual_entities} does not match {expected_entities}"

    print("Test passed.")

# Run the test
test_extract_named_entities()