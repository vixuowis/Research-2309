# requirements_file --------------------

!pip install -U flair

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_named_entities(text):
    """
    Extract named entities from the given text snippet using Flair's NER model.

    Parameters:
    text (str): A string containing the text to be analyzed for named entities.

    Returns:
    entities (list): A list of named entities found in the text with their types.
    """
    # Load the pre-trained NER model
    tagger = SequenceTagger.load('flair/ner-english-ontonotes')

    # Create a Sentence object with the input text
    sentence = Sentence(text)

    # Predict named entities in the text
    tagger.predict(sentence)

    # Extract entities and return them
    return [(str(entity), entity.tag) for entity in sentence.get_spans('ner')]

# test_function_code --------------------

def test_extract_named_entities():
    print("Testing extract_named_entities function.")

    # Test case 1: Check known entities
    text1 = "On June 7th, Jane Smith visited the Empire State Building in New York with an entry fee of 35 dollars."
    entities1 = extract_named_entities(text1)
    expected1 = [('June 7th', 'DATE'), ('Jane Smith', 'PERSON'), ('Empire State Building', 'FAC'), ('New York', 'GPE'), ('35 dollars', 'MONEY')]
    assert entities1 == expected1, f"Test case failed: Expected {expected1}, but got {entities1}"

    # Test case 2: Check empty string
    text2 = ""
    entities2 = extract_named_entities(text2)
    assert entities2 == [], "Test case failed: Expected [], but got non-empty list"

    # Test case 3: Check text without entities
    text3 = "This is a simple sentence without named entities."
    entities3 = extract_named_entities(text3)
    assert entities3 == [], "Test case failed: Expected [], but got non-empty list"

    print("All tests passed!")