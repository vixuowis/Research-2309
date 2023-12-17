# requirements_file --------------------

!pip install -U flair

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger

# function_code --------------------

def extract_entities_from_article(text):
    """
    Extract named entities from the text of a news article using a pre-trained NER model.

    Parameters:
    text (str): The text of the news article.

    Returns:
    list: A list of named entities found in the text.
    """
    tagger = SequenceTagger.load('flair/ner-english-ontonotes')
    sentence = Sentence(text)
    tagger.predict(sentence)
    return sentence.get_spans('ner')

# test_function_code --------------------

def test_extract_entities_from_article():
    print("Testing extract_entities_from_article function.")

    # Example news article text
    sample_text = "On September 1st, George Washington won 1 dollar in a lottery."

    # Expected entities
    expected_entities = [
        'DATE', 'PERSON', 'MONEY'
    ]  # This is a simplified version, actual output will be more complex

    # Extract entities from the sample text
    entities = extract_entities_from_article(sample_text)

    # Assert that the extracted entities match the expected entities
    assert all(entity.tag in expected_entities for entity in entities), "Failed to extract correct entities."
    print("Test passed.")

# Run the test
test_extract_entities_from_article()