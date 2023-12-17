# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_entities_from_article(article_text):
    """
    Extract named entities such as people, organizations, and locations from the given news article.

    Args:
        article_text (str): The text of the news article to analyze.

    Returns:
        list: A list of dictionaries with detected entities and corresponding labels.

    Raises:
        ValueError: If the article_text is empty or not provided.
    """
    if not article_text:
        raise ValueError('No article text provided')
    tokenizer = AutoTokenizer.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    model = AutoModelForTokenClassification.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    return nlp(article_text)

# test_function_code --------------------

def test_extract_entities_from_article():
    print("Testing started.")
    # Test case 1: Valid article text
    print("Testing case [1/2] started.")
    article_text = 'Nader Jokhadar had given Syria the lead with a well-struck header in the seventh minute.'
    result = extract_entities_from_article(article_text)
    expected_entities = ['PERSON', 'LOCATION']
    assert all(any(entity['entity_group'] == e for entity in result) for e in expected_entities), f"Test case [1/2] failed: Expected entities {expected_entities}, but got {result}"

    # Test case 2: Empty article text
    print("Testing case [2/2] started.")
    article_text = ''
    try:
        extract_entities_from_article(article_text)
        assert False, "Test case [2/2] failed: Expected ValueError for empty article text"
    except ValueError as e:
        assert str(e) == 'No article text provided', f"Test case [2/2] failed: Expected 'No article text provided', but got {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_entities_from_article()