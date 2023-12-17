# requirements_file --------------------

!pip install -U flair

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def identify_named_entities(german_text):
    """
    This function takes a German text as input and identifies named entities in the text,
    including person names, locations, organizations, and other names.

    :param german_text: A string containing the German text to be analyzed for named entities.
    :return: A list of tuples with the entity text and its type.
    """
    # Load the pre-trained Named Entity Recognition (NER) model for German
    tagger = SequenceTagger.load('flair/ner-german')

    # Create a Sentence object from the provided German text
    sentence = Sentence(german_text)

    # Use the predict() method to identify named entities in the sentence
    tagger.predict(sentence)

    # Extract the named entities and their types
    entities = [(entity.text, entity.get_label('ner').value) for entity in sentence.get_spans('ner')]
    
    # Return the list of entities
    return entities

# test_function_code --------------------

def test_identify_named_entities():
    print("Testing started.")
    
    # Test case 1: Check if person names are identified correctly
    sentence_with_person = "Angela Merkel war Bundeskanzlerin von Deutschland."
    expected_person = [('Angela Merkel', 'PER')]
    assert identify_named_entities(sentence_with_person) == expected_person, "Test case [1/3] failed: Person names not identified correctly."

    # Test case 2: Check if locations are identified correctly
    sentence_with_location = "Berlin ist die Hauptstadt von Deutschland."
    expected_location = [('Berlin', 'LOC')]
    assert identify_named_entities(sentence_with_location) == expected_location, "Test case [2/3] failed: Locations not identified correctly."

    # Test case 3: Check if organizations are identified correctly
    sentence_with_organization = "Die Vereinten Nationen haben ihren Sitz in New York."
    expected_organization = [('Die Vereinten Nationen', 'ORG'), ('New York', 'LOC')]
    assert identify_named_entities(sentence_with_organization) == expected_organization, "Test case [3/3] failed: Organizations not identified correctly."
    
    print("Testing finished.")

# Run the test function
test_identify_named_entities()