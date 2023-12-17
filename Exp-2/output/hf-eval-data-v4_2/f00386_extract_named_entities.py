# requirements_file --------------------

!pip install -U flair

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_named_entities(text):
    """
    Extract named entities from the provided text using pre-trained NER model.

    Args:
        text (str): The text from which to extract named entities.

    Returns:
        List[Span]: A list of named entity Spans found in the text.

    Raises:
        ValueError: If the input text is empty or not provided.

    """
    if not text:
        raise ValueError('Input text cannot be empty.')

    tagger = SequenceTagger.load('flair/ner-english-ontonotes-fast')
    sentence = Sentence(text)
    tagger.predict(sentence)
    return [entity for entity in sentence.get_spans('ner')]

# test_function_code --------------------

def test_extract_named_entities():
    print("Testing started.")
    # Use a text example to test the function
    text_example = "On September 1st, George Washington won 1 dollar."

    # Testing case 1: Check if named entities are correctly extracted
    print("Testing case [1/1] started.")
    entities = extract_named_entities(text_example)
    expected_entities = ['George Washington', 'September 1st', '1 dollar']
    assert [str(entity) for entity in entities] == expected_entities,
            f"Test case [1/1] failed: Expected {expected_entities}, got {[str(entity) for entity in entities]}"
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_named_entities()