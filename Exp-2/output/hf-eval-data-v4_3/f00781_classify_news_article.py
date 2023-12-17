# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_news_article(news_article):
    """
    Classify a given news article into predefined categories.

    Args:
        news_article (str): The news article to classify.

    Returns:
        dict: A dictionary with the classification results including labels and scores.

    Raises:
        ValueError: If the news article is not a string or is empty.
    """
    if not isinstance(news_article, str) or not news_article:
        raise ValueError('The news article must be a non-empty string.')
    candidate_labels = ['Politics', 'Sports', 'Technology', 'Business', 'Entertainment']
    classifier = pipeline('zero-shot-classification', model='typeform/squeezebert-mnli')
    return classifier(news_article, candidate_labels)

# test_function_code --------------------

def test_classify_news_article():
    print("Testing started.")
    # Sample news articles for testing
    sample_news_articles = [
        'Government passes new law to improve the economy.',
        'Local sports team wins championship after a thrilling final.',
        'New breakthrough in artificial intelligence technology.',
    ]

    # First test case
    print("Testing case [1/3] started.")
    result1 = classify_news_article(sample_news_articles[0])
    assert 'Politics' in result1['labels'], f"Test case [1/3] failed: {result1}"

    # Second test case
    print("Testing case [2/3] started.")
    result2 = classify_news_article(sample_news_articles[1])
    assert 'Sports' in result2['labels'], f"Test case [2/3] failed: {result2}"

    # Third test case
    print("Testing case [3/3] started.")
    try:
        classify_news_article('')
    except ValueError as e:
        assert str(e) == 'The news article must be a non-empty string.', f"Test case [3/3] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_news_article()