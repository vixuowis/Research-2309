# requirements_file --------------------

!pip install -U flair

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_named_entities(text):
    """
    Extract named entities from a given text using the pre-trained 'flair/ner-english' model.

    Parameters:
        text (str): The text from which to extract named entities.

    Returns:
        list: A list of named entities found in the text.
    """
    # Load the NER model
    tagger = SequenceTagger.load('flair/ner-english')

    # Create a Sentence object from the text
    sentence = Sentence(text)

    # Predict NER tags
    tagger.predict(sentence)

    # Get the named entities
    named_entities = [str(entity) for entity in sentence.get_spans('ner')]

    return named_entities

# test_function_code --------------------

def test_extract_named_entities():
    print("Testing extract_named_entities function.")

    # Test with a sample sentence
    sample_sentence = 'Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.'
    expected_entities = ['Apple-ORG', 'Steve Jobs-PER', 'Steve Wozniak-PER', 'Ronald Wayne-PER']

    # Run the function
    result = extract_named_entities(sample_sentence)

    # Check if the result matches the expected entities
    assert sorted(result) == sorted(expected_entities), f"Test failed: Expected {expected_entities} but got {result}"

    print("Test passed.")

# Call test function to verify correctness
test_extract_named_entities()