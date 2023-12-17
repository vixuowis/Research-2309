# requirements_file --------------------

!pip install -U flair

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_entities_from_email(customer_email_text):
    """
    Extracts entities from a customer email text using the Flair NER model.

    Args:
        customer_email_text (str): The text content of the customer's email.

    Returns:
        list: A list of entities recognized in the email text, each as a dictionary with 'text' and 'type'.

    Raises:
        ValueError: If customer_email_text is not a string.
    """
    if not isinstance(customer_email_text, str):
        raise ValueError('The email text must be a string.')

    # Load the Named Entity Recognition model
    tagger = SequenceTagger.load('flair/ner-english-ontonotes')
    # Create a Sentence object from the text of a customer email
    sentence = Sentence(customer_email_text)
    # Use the loaded NER model to predict entities in the sentence
    tagger.predict(sentence)

    # Extract the recognized entities
    entities = [{'text': entity.text, 'type': entity.tag} for entity in sentence.get_spans('ner')]

    return entities

# test_function_code --------------------

def test_extract_entities_from_email():
    print("Testing started.")
    # Example customer email text
    sample_email_text = 'Hello, my name is John Doe, I work at Acme Corp. located in New York. Here is my phone number: 123-456-7890.'

    # Expected entities
    expected_entities = [
        {'text': 'John Doe', 'type': 'PERSON'},
        {'text': 'Acme Corp.', 'type': 'ORG'},
        {'text': 'New York', 'type': 'GPE'},
        {'text': '123-456-7890', 'type': 'CARDINAL'}
    ]

    # Test case 1: Valid email text
    print("Testing case [1/1] started.")
    actual_entities = extract_entities_from_email(sample_email_text)
    assert actual_entities == expected_entities, f"Test case [1/1] failed: expected {expected_entities}, got {actual_entities}"
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_entities_from_email()