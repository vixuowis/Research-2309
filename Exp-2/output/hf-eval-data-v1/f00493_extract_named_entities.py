from flair.data import Sentence
from flair.models import SequenceTagger

def extract_named_entities(text):
    '''
    This function extracts the named entities from a given text snippet using the 'flair/ner-english-ontonotes' pre-trained model.
    
    Parameters:
    text (str): The text snippet from which to extract the named entities.
    
    Returns:
    list: A list of named entities extracted from the text.
    '''
    # Load the 'flair/ner-english-ontonotes' pre-trained model as a SequenceTagger
    tagger = SequenceTagger.load('flair/ner-english-ontonotes')
    
    # Pass the input text to the Sentence constructor to create a Sentence object
    sentence = Sentence(text)
    
    # Use the loaded SequenceTagger's predict method with the Sentence object as an argument to generate the named entities
    tagger.predict(sentence)
    
    # Iterate through the sentence's named entity spans (found using the get_spans method) and return them
    return [entity for entity in sentence.get_spans('ner')]