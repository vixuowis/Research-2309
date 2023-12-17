# requirements_file --------------------

!pip install -U flair

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger


# function_code --------------------

def detect_named_entities(text):
    """
    Detect named entities in a given text using a pre-trained NER model.

    Args:
        text (str): The text to be analyzed for named entities.

    Returns:
        List: A list of named entities found in the text, each represented as a dictionary containing the entity's text, start position, end position, and type.

    Raises:
        ValueError: If the input text is empty or not a string.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("Input text must be a non-empty string.")

    # Load the NER tagger
    tagger = SequenceTagger.load('flair/ner-english-ontonotes-large')

    # Create a Sentence object
    sentence = Sentence(text)

    # Predict NER tags
    tagger.predict(sentence)

    # Extract entities
    entities = [{
        'text': entity.text,
        'start_pos': entity.start_pos,
        'end_pos': entity.end_pos,
        'type': entity.get_label('ner').value
    } for entity in sentence.get_spans('ner')]

    return entities


# test_function_code --------------------

def test_detect_named_entities():
    print("Testing started.")

    # Test case 1: A sentence with named entities
    print("Testing case [1/3] started.")
    sentence1 = "Jon went to Paris with his friend Alex on September 20th, 2022."
    entities1 = detect_named_entities(sentence1)
    assert entities1, f"Test case [1/3] failed: Expected named entities, got {entities1}"

    # Test case 2: A sentence without named entities
    print("Testing case [2/3] started.")
    sentence2 = "Hello world!"
    entities2 = detect_named_entities(sentence2)
    assert not entities2, f"Test case [2/3] failed: Expected no entities, got {entities2}"

    # Test case 3: An empty string
    print("Testing case [3/3] started.")
    try:
        detect_named_entities("")
    except ValueError as e:
        assert str(e) == "Input text must be a non-empty string.", f"Test case [3/3] failed: {e}"

    print("Testing finished.")


# call_test_function_line --------------------

test_detect_named_entities()