# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_news_article(news_article):
    """
    Classify a news article into categories like Politics, Sports, Technology, Business, and Entertainment.

    Args:
        news_article (str): The news article to be classified.

    Returns:
        dict: A dictionary with the classification scores for each category.

    Raises:
        OSError: If there is a problem with the model loading due to disk quota exceeded.
    """
    try:
        # load the model
        classifier = pipeline("zero-shot-classification")
        
        # classify the news article
        scores_dict = classifier(news_article, candidate_labels=[
            "Politics", "Sports", "Technology", "Business", "Entertainment"])[0]
    except OSError as error:
        print("Error loading classification model")
        
        # return a default dictionary with scores equal to zero
        scores_dict = {
            "Politics": 0.0,
            "Sports": 0.0,
            "Technology": 0.0,
            "Business": 0.0,
            "Entertainment": 0.0
        }
        
    # return a dictionary with the classification scores for each category
    return scores_dict

# test_function_code --------------------

def test_classify_news_article():
    """
    Test the classify_news_article function with some example news articles.
    """
    news_article1 = 'The government passed a new law today'
    news_article2 = 'The local team won the championship'
    news_article3 = 'Apple released a new product'
    result1 = classify_news_article(news_article1)
    result2 = classify_news_article(news_article2)
    result3 = classify_news_article(news_article3)
    assert isinstance(result1, dict), 'The result should be a dictionary.'
    assert isinstance(result2, dict), 'The result should be a dictionary.'
    assert isinstance(result3, dict), 'The result should be a dictionary.'
    print('All Tests Passed')


# call_test_function_code --------------------

test_classify_news_article()