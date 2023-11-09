from flair.data import Sentence
from flair.models import SequenceTagger

def extract_named_entities(text):
    '''
    This function extracts named entities from a given text using the 'flair/ner-english-ontonotes-fast' model.
    
    Parameters:
    text (str): The text from which to extract named entities.
    
    Returns:
    list: A list of named entities extracted from the text.
    '''
    # Load the 'flair/ner-english-ontonotes-fast' NER model
    tagger = SequenceTagger.load('flair/ner-english-ontonotes-fast')
    
    # Prepare the input text by converting it into a Sentence object
    sentence = Sentence(text)
    
    # Pass this Sentence object to the tagger.predict() method to obtain the NER annotations
    tagger.predict(sentence)
    
    # Extract the tagged entities in a structured format
    named_entities = [entity for entity in sentence.get_spans('ner')]
    
    return named_entities