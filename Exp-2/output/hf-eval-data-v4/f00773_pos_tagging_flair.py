# requirements_file --------------------

!pip install -U flair

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def pos_tagging_flair(text):
    """
    Analyzes the part-of-speech tags for an input text using the Flair 'pos-english' model.

    :param text: str - The text string to analyze
    :return: List[Tuple[str, str]] - List of tuples where each tuple represents a word and its corresponding POS tag
    """
    tagger = SequenceTagger.load('flair/pos-english')
    sentence = Sentence(text)
    tagger.predict(sentence)

    return [(entity.text, entity.get_label('pos').value) for entity in sentence.get_spans('pos')]

# test_function_code --------------------

def test_pos_tagging_flair():
    print("Testing pos_tagging_flair() function.")

    # Example sentence from the provided data.
    sentence = 'The quick brown fox jumps over the lazy dog.'
    expected_tags = [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN')]
    pos_tags = pos_tagging_flair(sentence)

    assert pos_tags == expected_tags, f"Test failed: {pos_tags}"

    print("All tests passed!")

# Calling the test function
test_pos_tagging_flair()