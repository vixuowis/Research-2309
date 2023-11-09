from flair.data import Sentence
from flair.models import SequenceTagger

def identify_entities(diary_entry_text):
    '''
    This function identifies the names of people and locations mentioned in a given diary entry.
    It uses the 'flair/ner-english' model from Hugging Face Transformers for Named Entity Recognition.
    
    Parameters:
    diary_entry_text (str): The text of the diary entry.
    
    Returns:
    list: A list of recognized names of people and locations.
    '''
    # Load the 'flair/ner-english' model
    tagger = SequenceTagger.load('flair/ner-english')
    
    # Create a 'Sentence' object from the diary entry text
    diary_entry = Sentence(diary_entry_text)
    
    # Predict NER tags for each token in the sentence
    tagger.predict(diary_entry)
    
    # Initialize an empty list to store the recognized names of people and locations
    entities = []
    
    # Iterate over the entities and add the recognized names of people and locations to the list
    for entity in diary_entry.get_spans('ner'):
        if entity.tag == 'PER' or entity.tag == 'LOC':
            entities.append(str(entity))
    
    return entities