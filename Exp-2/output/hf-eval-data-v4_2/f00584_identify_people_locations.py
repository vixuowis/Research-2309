# requirements_file --------------------

!pip install -U flair

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def identify_people_locations(text):
    """
    Identify and print the names of people and locations mentioned in the text using NER model.

    Args:
        text (str): The text to analyze for named entities.

    Returns:
        list: A list of tuples containing found entities and their types (either 'PER' for person or 'LOC' for location).

    Raises:
        ValueError: If the input text is not provided or empty.
    """
    if not text:
        raise ValueError('Input text must be provided.')

    tagger = SequenceTagger.load('flair/ner-english')
    sentence = Sentence(text)
    tagger.predict(sentence)

    people_locations = [(entity.text, entity.tag) for entity in sentence.get_spans('ner') if entity.tag in ('PER', 'LOC')]
    return people_locations

# test_function_code --------------------

def test_identify_people_locations():
    print("Testing started.")

    # Test case 1: Check if it correctly identifies a person's name
    print("Testing case [1/3] started.")
    result1 = identify_people_locations('Yesterday, I met with Mark in Berlin.')
    assert ('Mark', 'PER') in result1, f"Test case [1/3] failed: {result1}"

    # Test case 2: Check if it correctly identifies a location name
    print("Testing case [2/3] started.")
    result2 = identify_people_locations('Our office is located at the Silicon Valley.')
    assert ('Silicon Valley', 'LOC') in result2, f"Test case [2/3] failed: {result2}"

    # Test case 3: Check if it correctly identifies both entities in a sentence
    print("Testing case [3/3] started.")
    result3 = identify_people_locations('Eva and Jacob went to Paris for a conference.')
    assert ('Eva', 'PER') in result3 and ('Paris', 'LOC') in result3, f"Test case [3/3] failed: {result3}"
    print("Testing finished.")

# call_test_function_line --------------------

test_identify_people_locations()