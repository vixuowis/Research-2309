# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def classify_news_articles(news_article, candidate_labels=['technology', 'sports', 'politics']):
    """
    Classify a news article into categories using zero-shot classification.

    Args:
        news_article (str): The content of the news article to classify.
        candidate_labels (list, optional): A list of strings representing the candidate categories. Defaults to ['technology', 'sports', 'politics'].

    Returns:
        dict: A dictionary where keys are the candidate labels and values are the corresponding scores.
    """
    cross_encoder = CrossEncoder('cross-encoder/nli-roberta-base')
    scores = cross_encoder.predict([{'sentence1': news_article, 'sentence2': label} for label in candidate_labels])
    return dict(zip(candidate_labels, scores))

# test_function_code --------------------

def test_classify_news_articles():
    """Tests for the `classify_news_articles` function"""
    article1 = 'Apple just announced the newest iPhone X'
    article2 = 'The government has passed a new law'
    article3 = 'The local team won the championship'
    assert classify_news_articles(article1)['technology'] > classify_news_articles(article1)['sports']
    assert classify_news_articles(article2)['politics'] > classify_news_articles(article2)['technology']
    assert classify_news_articles(article3)['sports'] > classify_news_articles(article3)['politics']
    print('All Tests Passed')

# call_test_function_code --------------------

test_classify_news_articles()