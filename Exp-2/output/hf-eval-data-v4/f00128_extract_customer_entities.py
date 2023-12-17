# requirements_file --------------------

!pip install -U flair

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_customer_entities(customer_email_text):
    """
    Extract entities such as names, organizations, and dates from customer email text.

    :param customer_email_text: Text content of the customer's email.
    :return: List of entities identified in the text.
    """
    # Load the pre-trained NER model
    tagger = SequenceTagger.load('flair/ner-english-ontonotes')
    # Create a Sentence object
    sentence = Sentence(customer_email_text)
    # Predict the entities
    tagger.predict(sentence)
    # Return the list of entities
    return [str(entity) for entity in sentence.get_spans('ner')]

# test_function_code --------------------

def test_extract_customer_entities():
    print("Testing extract_customer_entities started.")
    sample_email = "Dear team, I visited your New York office on April 1st and talked to George."
    # Expected entities
    expected_entities = ['George', 'New York', 'April 1st']
    # Extract entities
    entities = extract_customer_entities(sample_email)
    # Test if the extracted entities match the expected
    assert sorted(entities) == sorted(expected_entities), f"Test failed: extracted entities {entities} do not match expected {expected_entities}"
    print("Test extract_customer_entities finished successfully.")