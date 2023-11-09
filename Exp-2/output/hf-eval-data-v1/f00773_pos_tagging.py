from flair.data import Sentence
from flair.models import SequenceTagger

def pos_tagging(input_text):
    """
    Function to perform part-of-speech tagging on a given text using the 'flair/pos-english' model.

    Args:
        input_text (str): The text to be tagged.

    Returns:
        List of tuples: Each tuple contains a word from the input text and its corresponding POS tag.
    """
    tagger = SequenceTagger.load('flair/pos-english')
    sentence = Sentence(input_text)
    tagger.predict(sentence)
    pos_tags = [(entity.text, entity.tag) for entity in sentence.get_spans('pos')]
    return pos_tags