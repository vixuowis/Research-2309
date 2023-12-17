# requirements_file --------------------

!pip install -U flair

# function_import --------------------

from flair.data import Sentence
from flair.models import SequenceTagger


# function_code --------------------

def extract_entities_from_article(article_text):
    """
    Extract named entities from a news article using the Flair library.

    Args:
        article_text (str): Text of the news article to analyze.

    Returns:
        list: A list of entities found in the article, each represented as a Flair Span.

    Raises:
        ValueError: If the provided text is empty.
    """
    if not article_text:
        raise ValueError('The article text must not be empty.')

    # Load the pre-trained NER model
    tagger = SequenceTagger.load('flair/ner-english-ontonotes')

    # Create a Sentence object for the input text
    sentence = Sentence(article_text)

    # Predict named entities in the text
    tagger.predict(sentence)

    # Extract entities
    entities = sentence.get_spans('ner')
    return entities


# test_function_code --------------------

def test_extract_entities_from_article():
    print("Testing started.")
    # Sample text for the test case
    sample_article = 'On September 1st, President Abraham Lincoln delivers the Gettysburg Address.'

    # Test case 1: Validate entity extraction
    print("Testing case [1/2] started.")
    entities = extract_entities_from_article(sample_article)
    assert len(entities) > 0, f"Test case [1/2] failed: No entities found."

    # Test case 2: Validate exception for empty text
    print("Testing case [2/2] started.")
    try:
        extract_entities_from_article('')
        assert False, f"Test case [2/2] failed: ValueError not raised on empty text."
    except ValueError:
        pass  # Expected behavior
    print("Testing finished.")


# call_test_function_line --------------------

test_extract_entities_from_article()