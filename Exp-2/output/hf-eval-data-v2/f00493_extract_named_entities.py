# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_named_entities(text):
    """
    Extract the named entities from a given text snippet using the 'flair/ner-english-ontonotes' pre-trained model.

    Args:
        text (str): The text snippet to extract named entities from.

    Returns:
        List[Tuple[str, str]]: A list of tuples where each tuple represents a named entity. The first element of the tuple is the entity and the second element is the entity type.
    """
    tagger = SequenceTagger.load('flair/ner-english-ontonotes')
    sentence = Sentence(text)
    tagger.predict(sentence)
    entities = [(entity.text, entity.tag) for entity in sentence.get_spans('ner')]
    return entities

# test_function_code --------------------

def test_extract_named_entities():
    """
    Test the extract_named_entities function.
    """
    test_text = 'On June 7th, Jane Smith visited the Empire State Building in New York with an entry fee of 35 dollars.'
    expected_output = [('June 7th', 'DATE'), ('Jane Smith', 'PERSON'), ('the Empire State Building', 'FAC'), ('New York', 'GPE'), ('35 dollars', 'MONEY')]
    assert set(extract_named_entities(test_text)) == set(expected_output)

# call_test_function_code --------------------

test_extract_named_entities()