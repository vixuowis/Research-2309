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
    
    # Load the fine-tuned model
    try:
        classifier = pipeline("zero-shot-classification", model="Sahajtomar/GPT-J-FinFine")
    except OSError as e:
        raise Exception(f"The following error was raised during loading of the classification model: {e}")
        
    # Classify the news article
    try:
        results = classifier(news_article, candidate_labels=["Politics", "Sports", "Technology", "Business", "Entertainment"])
    except OSError as e:
        raise Exception(f"The following error was raised during classification of the news article: {e}")
    
    # Return the category classification scores 
    scores = dict(zip(results["labels"], results["scores"]))
    return scores

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