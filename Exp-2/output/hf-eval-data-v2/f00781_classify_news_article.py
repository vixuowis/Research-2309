# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_news_article(news_article):
    """
    Classify a news article into categories like Politics, Sports, Technology, Business, and Entertainment.

    Args:
        news_article (str): The text of the news article to classify.

    Returns:
        dict: A dictionary with the labels as keys and their corresponding scores as values.
    """
    classifier = pipeline('zero-shot-classification', model='typeform/squeezebert-mnli')
    candidate_labels = ['Politics', 'Sports', 'Technology', 'Business', 'Entertainment']
    result = classifier(news_article, candidate_labels)
    return result

# test_function_code --------------------

def test_classify_news_article():
    """
    Test the classify_news_article function with a sample news article.
    """
    news_article = 'The tech giant Google is planning to buy the startup for 1 billion dollars.'
    result = classify_news_article(news_article)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'Technology' in result, 'The result should contain the label Technology.'

# call_test_function_code --------------------

test_classify_news_article()