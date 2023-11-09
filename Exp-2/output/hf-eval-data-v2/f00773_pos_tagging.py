# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def pos_tagging(input_text):
    """
    This function uses the 'flair/pos-english' model to perform part-of-speech tagging on the input text.

    Args:
        input_text (str): The text to be analyzed.

    Returns:
        List of tuples: A list of tuples where each tuple represents a word and its corresponding POS tag.
    """
    tagger = SequenceTagger.load('flair/pos-english')
    sentence = Sentence(input_text)
    tagger.predict(sentence)
    pos_tags = [(entity.text, entity.tag) for entity in sentence.get_spans('pos')]
    return pos_tags

# test_function_code --------------------

def test_pos_tagging():
    """
    This function tests the 'pos_tagging' function with a sample sentence.
    """
    input_text = 'The quick brown fox jumps over the lazy dog.'
    expected_output = [('The', 'DT'), ('quick', 'JJ'), ('brown', 'NN'), ('fox', 'NN'), ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN'), ('.', '.')]
    assert pos_tagging(input_text) == expected_output

# call_test_function_code --------------------

test_pos_tagging()