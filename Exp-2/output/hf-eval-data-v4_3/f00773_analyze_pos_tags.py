# requirements_file --------------------

import subprocess

requirements = ["flair"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def analyze_pos_tags(text):
    """
    Analyze the part-of-speech tags of a given text using a pre-trained model.

    Args:
        text (str): The text to be analyzed.

    Returns:
        List[Tuple[str, str]]: A list of tuples with the word and its part-of-speech tag.

    Raises:
        ValueError: If the input text is empty.

    """
    if not text:
        raise ValueError('Input text cannot be empty')

    tagger = SequenceTagger.load('flair/pos-english')
    sentence = Sentence(text)
    tagger.predict(sentence)

    pos_tags = [(entity.text, entity.get_label('pos').value) for entity in sentence.get_spans('pos')]
    return pos_tags

# test_function_code --------------------

def test_analyze_pos_tags():
    print("Testing started.")

    # Test case 1: Proper text input
    print("Testing case [1/2] started.")
    result = analyze_pos_tags('The quick brown fox jumps over the lazy dog.')
    expected = [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN')]
    assert result == expected, f"Test case [1/2] failed: {result}!="

    # Test case 2: Empty text input
    print("Testing case [2/2] started.")
    try:
        analyze_pos_tags('')
        assert False, "Test case [2/2] failed: ValueError not raised on empty input!"
    except ValueError as e:
        assert str(e) == 'Input text cannot be empty', f"Test case [2/2] failed: Unexpected ValueError message: {str(e)}!"

    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_pos_tags()