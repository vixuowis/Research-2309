# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_news_headlines(headline: str, candidate_labels: list) -> dict:
    """
    Classify news headlines into categories using a zero-shot classifier.

    Args:
        headline (str): The news headline to classify.
        candidate_labels (list): The list of categories to classify the headline into.

    Returns:
        dict: The classification results.

    Raises:
        ValueError: If the headline is not a string or candidate_labels is not a list.
    """
    
    if (not isinstance(headline, str) 
        and not isinstance(candidate_labels, list)):
            raise TypeError(f'The `headline` argument must be of type string, and the `candidate_labels` arg must be a list.')
            
    # load zero-shot classification pipeline
    classifier = pipeline('zero-shot-classification', device=-1)
    
    result = {}
    
    try:
        result = classifier(headline, 
                            candidate_labels=candidate_labels,
                            multi_label=True)
        
    except Exception as e:
        print('There was an error with the classification task.')
        
        # log exception message
        logger.exception("An exception occurred in the classify_news_headlines() method.")
    
    return result

# test_function_code --------------------

def test_classify_news_headlines():
    """Tests for the classify_news_headlines function"""
    headline1 = 'Apple just announced the newest iPhone X'
    candidate_labels1 = ['technology', 'sports', 'politics']
    result1 = classify_news_headlines(headline1, candidate_labels1)
    assert isinstance(result1, dict), 'Result must be a dictionary'
    assert 'labels' in result1, 'Result dictionary must contain labels'
    assert 'scores' in result1, 'Result dictionary must contain scores'

    headline2 = 'The Lakers won their game last night'
    candidate_labels2 = ['technology', 'sports', 'politics']
    result2 = classify_news_headlines(headline2, candidate_labels2)
    assert isinstance(result2, dict), 'Result must be a dictionary'
    assert 'labels' in result2, 'Result dictionary must contain labels'
    assert 'scores' in result2, 'Result dictionary must contain scores'

    print('All Tests Passed')


# call_test_function_code --------------------

test_classify_news_headlines()