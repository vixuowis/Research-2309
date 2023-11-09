from flair.data import Sentence
from flair.models import SequenceTagger

def detect_entities(input_sentence):
    '''
    This function detects named entities in a given sentence using the Flair NER model.
    
    Parameters:
    input_sentence (str): The sentence in which to detect named entities.
    
    Returns:
    list: A list of detected entities.
    '''
    # Load the Flair NER model
    tagger = SequenceTagger.load('flair/ner-english-ontonotes-large')
    
    # Create a new Sentence object with the input text
    sentence = Sentence(input_sentence)
    
    # Use the NER model to predict tags
    tagger.predict(sentence)
    
    # Extract the detected entities
    entities = [entity for entity in sentence.get_spans('ner')]
    
    return entities