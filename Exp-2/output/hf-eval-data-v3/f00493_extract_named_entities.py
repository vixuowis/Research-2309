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
    tagger = SequenceTagger.load('flair/ner-english-ontonotes')
    sentence = Sentence(text)
    tagger.predict(sentence)
    entities = [entity for entity in sentence.get_spans('ner')]
    return entities

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