# requirements_file --------------------

!pip install -U flair

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_people_locations(text):
    """
    Extracts names of people and locations from the given text.

    Args:
    text (str): The text from which to extract entity names.

    Returns:
    list: A list of dictionaries with entity name and its type (person or location).
    """
    entities = []
    # load the NER tagger model
tagger = SequenceTagger.load('flair/ner-english')
    # create a sentence
text_to_analyze = Sentence(text)
    # predict NER tags
tagger.predict(text_to_analyze)
    # iterate over entities and extract names of people and locations
for entity in text_to_analyze.get_spans('ner'):
        if entity.tag in ['PER', 'LOC']:
            entities.append({'text': entity.text, 'type': 'person' if entity.tag == 'PER' else 'location'})
    return entities

# test_function_code --------------------

def test_extract_people_locations():
    print("Testing started.")
    # Test case 1: Basic sentence with one person and one location.
    text1 = 'George Washington went to Washington.'
    expected1 = [{'text': 'George Washington', 'type': 'person'}, {'text': 'Washington', 'type': 'location'}]
    result1 = extract_people_locations(text1)
    assert result1 == expected1, f"Test case failed: {result1} != {expected1}"

    # Test case 2: Sentence with no person or location names.
    text2 = 'There is nothing to extract here.'
    expected2 = []
    result2 = extract_people_locations(text2)
    assert result2 == expected2, f"Test case failed: {result2} != {expected2}"

    # Test case 3: Complex sentence with multiple entities.
    text3 = 'Mark Zuckerberg was born in New York and founded Facebook.'
    expected3 = [{'text': 'Mark Zuckerberg', 'type': 'person'}, {'text': 'New York', 'type': 'location'}]
    result3 = extract_people_locations(text3)
    assert result3 == expected3, f"Test case failed: {result3} != {expected3}"

    print("Testing finished.")