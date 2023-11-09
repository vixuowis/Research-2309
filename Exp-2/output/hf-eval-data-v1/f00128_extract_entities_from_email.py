from flair.data import Sentence
from flair.models import SequenceTagger

def extract_entities_from_email(customer_email_text):
    '''
    This function uses the Flair NER model 'flair/ner-english-ontonotes' to identify and classify entities in a given text.
    The model is capable of predicting 18 tags such as cardinal value, date value, event name, building name, geo-political entity, language name, law name, location name, money name, affiliation, ordinal value, organization name, percent value, person name, product name, quantity value, time value, and name of work of art.
    
    Args:
    customer_email_text (str): The text from a customer email.
    
    Returns:
    list: A list of recognized entities in the text.
    '''
    tagger = SequenceTagger.load('flair/ner-english-ontonotes')
    sentence = Sentence(customer_email_text)
    tagger.predict(sentence)
    
    entities = [entity for entity in sentence.get_spans('ner')]
    return entities