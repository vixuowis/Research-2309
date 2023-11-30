# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_review_sentiment(review_text):
    """
    Analyze the sentiment of a product review using a pre-trained model.

    Args:
        review_text (str): The text of the product review.

    Returns:
        dict: The sentiment analysis result, including the label (number of stars) and the score.
    """

    # Initialize model --------------------
    
    sentiment = pipeline("sentiment-analysis")
    
    # Get result --------------------
    
    result = sentiment(review_text)[0]
    result["label"] = 1 if result["score"] >= 0.5 else 5
    
    return result

# test_function_code --------------------

def test_analyze_review_sentiment():
    """
    Test the analyze_review_sentiment function.
    """
    # Test with a positive review
    result = analyze_review_sentiment('I love this product!')
    assert result['label'] in ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
    assert 0 <= result['score'] <= 1

    # Test with a negative review
    result = analyze_review_sentiment('I hate this product!')
    assert result['label'] in ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
    assert 0 <= result['score'] <= 1

    # Test with a neutral review
    result = analyze_review_sentiment('This product is okay.')
    assert result['label'] in ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
    assert 0 <= result['score'] <= 1

    return 'All Tests Passed'


# call_test_function_code --------------------

test_analyze_review_sentiment()