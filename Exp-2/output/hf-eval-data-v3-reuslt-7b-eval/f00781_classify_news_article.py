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
    
    # Instantiate classifier pipeline
    classifier = pipeline("zero-shot-classification", device=0)
    
    # Run prediction
    try:
        results = classifier(news_article, candidate_labels=["Politics", "Sports", "Technology", \
                            "Business", "Entertainment"], multi_label=False)
        
    except OSError as e:
        print("Disk quota exceeded")
        raise(e)
    
    # Return results in a dictionary format
    return results

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