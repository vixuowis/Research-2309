# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def identify_entities(text):
    """
    Identify the entities like persons, locations, organizations, and other names in a given German text.

    Args:
        text (str): The German text to identify entities from.

    Returns:
        list: A list of entities identified from the text. Each entity is represented as a dictionary with 'text', 'start_pos', 'end_pos', 'type' keys.
    """
    tagger = SequenceTagger.load('flair/ner-german')
    sentence = Sentence(text)
    tagger.predict(sentence)
    entities = [{'text': entity.text, 'start_pos': entity.start_pos, 'end_pos': entity.end_pos, 'type': entity.tag} for entity in sentence.get_spans('ner')]
    return entities

# test_function_code --------------------

def test_identify_entities():
    assert identify_entities('George Washington ging nach Washington') == [{'text': 'George Washington', 'start_pos': 0, 'end_pos': 16, 'type': 'PER'}, {'text': 'Washington', 'start_pos': 22, 'end_pos': 32, 'type': 'LOC'}]
    assert identify_entities('Ich arbeite bei Google in Mountain View') == [{'text': 'Google', 'start_pos': 14, 'end_pos': 20, 'type': 'ORG'}, {'text': 'Mountain View', 'start_pos': 24, 'end_pos': 36, 'type': 'LOC'}]
    assert identify_entities('Angela Merkel ist die Bundeskanzlerin von Deutschland') == [{'text': 'Angela Merkel', 'start_pos': 0, 'end_pos': 13, 'type': 'PER'}, {'text': 'Bundeskanzlerin', 'start_pos': 18, 'end_pos': 33, 'type': 'MISC'}, {'text': 'Deutschland', 'start_pos': 38, 'end_pos': 49, 'type': 'LOC'}]
    return 'All Tests Passed'

# call_test_function_code --------------------

test_identify_entities()