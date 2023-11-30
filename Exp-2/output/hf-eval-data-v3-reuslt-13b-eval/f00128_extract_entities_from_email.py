# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_entities_from_email(customer_email_text):
    """
    Extract entities from customer email text using Flair's NER model.

    Args:
        customer_email_text (str): The text of the customer email.

    Returns:
        list: A list of recognized entities in the email text.
    """

    # Initialize the tagger and sentence.
    flair_tagger = SequenceTagger.load("ner")
    email_sentence = Sentence(customer_email_text)

    # Run the named entity recognition on sentence.
    flair_tagger.predict(email_sentence)
    
    return [ent for ent in email_sentence.to_dict(tag_type='ner')['entities']]

# test_function_code --------------------

def test_extract_entities_from_email():
    """
    Test the function extract_entities_from_email.
    """
    email1 = 'On September 1st George Washington won 1 dollar.'
    email2 = 'The meeting will be held at the Google headquarters on 25th December.'
    email3 = 'I bought an iPhone for $699.'
    assert len(extract_entities_from_email(email1)) > 0
    assert len(extract_entities_from_email(email2)) > 0
    assert len(extract_entities_from_email(email3)) > 0
    print('All Tests Passed')


# call_test_function_code --------------------

test_extract_entities_from_email()