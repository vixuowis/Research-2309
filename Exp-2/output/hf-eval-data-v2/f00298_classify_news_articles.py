# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def classify_news_articles(news_article, candidate_labels=['technology', 'sports', 'politics']):
    """
    Classify news articles into their respective categories using zero-shot classification.

    Args:
        news_article (str): The content of the news article to be classified.
        candidate_labels (list, optional): The list of categories into which the article can be classified. Defaults to ['technology', 'sports', 'politics'].

    Returns:
        dict: A dictionary where keys are the candidate labels and values are the corresponding scores.
    """
    cross_encoder = CrossEncoder('cross-encoder/nli-roberta-base')
    scores = cross_encoder.predict([{'sentence1': news_article, 'sentence2': label} for label in candidate_labels])
    return dict(zip(candidate_labels, scores))

# test_function_code --------------------

def test_classify_news_articles():
    """
    Test the function classify_news_articles.
    """
    news_article = 'Apple just announced the newest iPhone X'
    candidate_labels = ['technology', 'sports', 'politics']
    result = classify_news_articles(news_article, candidate_labels)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert set(candidate_labels) == set(result.keys()), 'The keys of the result should be the candidate labels.'
    assert all(isinstance(score, float) for score in result.values()), 'The values of the result should be floats.'

# call_test_function_code --------------------

test_classify_news_articles()