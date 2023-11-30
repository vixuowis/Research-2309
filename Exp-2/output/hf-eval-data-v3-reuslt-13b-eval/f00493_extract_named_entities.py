# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_named_entities(text):
    '''
    Extract the named entities from a given text snippet.

    Args:
        text (str): The text snippet to extract named entities from.

    Returns:
        list: A list of named entities found in the text.
    '''

    # load flair's pretrained NER tagger
    ner = SequenceTagger.load('ner')
    
    # create a sentence object to find named entities from it using Flair
    # this step will return an array of objects with start_pos, end_post and text
    sentence = Sentence(text)

    # run the NER tagger over the given text
    ner.predict(sentence)
    
    # get all entities found in the provided text
    entities = sentence.get_spans('ner')
    names = [ent.text for ent in entities]
    
    return names

# test_function_code --------------------

def test_extract_named_entities():
    '''
    Test the extract_named_entities function.
    '''
    assert len(extract_named_entities('On June 7th, Jane Smith visited the Empire State Building in New York with an entry fee of 35 dollars.')) > 0
    assert len(extract_named_entities('George Washington was the first president of the United States.')) > 0
    assert len(extract_named_entities('The Eiffel Tower is located in Paris.')) > 0
    return 'All Tests Passed'


# call_test_function_code --------------------

test_extract_named_entities()