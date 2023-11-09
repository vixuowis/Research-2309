# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_entities_from_email(customer_email_text):
    """
    Extracts entities from a given customer email text using the Flair NER model.

    Args:
        customer_email_text (str): The text of the customer email.

    Returns:
        A list of tuples where each tuple represents an entity. Each tuple contains the entity value and its type.
    """
    tagger = SequenceTagger.load('flair/ner-english-ontonotes')
    sentence = Sentence(customer_email_text)
    tagger.predict(sentence)
    entities = [(entity.text, entity.tag) for entity in sentence.get_spans('ner')]
    return entities

# test_function_code --------------------

def test_extract_entities_from_email():
    """
    Tests the function extract_entities_from_email.
    """
    test_email = 'On September 1st George Washington won 1 dollar.'
    expected_output = [('September 1st', 'DATE'), ('George Washington', 'PERSON'), ('1 dollar', 'MONEY')]
    assert set(extract_entities_from_email(test_email)) == set(expected_output)

# call_test_function_code --------------------

test_extract_entities_from_email()