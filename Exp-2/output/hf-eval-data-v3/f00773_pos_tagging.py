# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def pos_tagging(input_text):
    '''
    Function to perform part-of-speech tagging on a given text using Flair's SequenceTagger.

    Args:
        input_text (str): The text to be tagged.

    Returns:
        List of tuples: Each tuple contains a word from the input text and its corresponding POS tag.
    '''
    tagger = SequenceTagger.load('flair/pos-english')
    sentence = Sentence(input_text)
    tagger.predict(sentence)
    pos_tags = [(entity.text, entity.tag) for entity in sentence.get_spans('pos')]
    return pos_tags

# test_function_code --------------------

def test_pos_tagging():
    '''
    Function to test the pos_tagging function.
    '''
    assert pos_tagging('The quick brown fox jumps over the lazy dog.') == [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN'), ('.', '.')]
    assert pos_tagging('I love Berlin.') == [('I', 'PRP'), ('love', 'VBP'), ('Berlin', 'NNP'), ('.', '.')]
    assert pos_tagging('This is a test sentence.') == [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('test', 'NN'), ('sentence', 'NN'), ('.', '.')]
    return 'All Tests Passed'

# call_test_function_code --------------------

test_pos_tagging()