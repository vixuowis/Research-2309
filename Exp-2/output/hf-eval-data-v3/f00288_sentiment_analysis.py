# function_import --------------------

from transformers import pipeline

# function_code --------------------

def sentiment_analysis(review: str) -> str:
    '''
    This function uses the Hugging Face Transformers library to perform sentiment analysis on a given review.
    
    Args:
        review (str): The review text that needs to be analyzed.
    
    Returns:
        str: The sentiment of the review, either 'POSITIVE' or 'NEGATIVE'.
    
    Raises:
        ValueError: If the review is not a string.
    '''
    if not isinstance(review, str):
        raise ValueError('Review must be a string.')
    sentiment_analysis_model = pipeline('text-classification', model='Seethal/sentiment_analysis_generic_dataset')
    result = sentiment_analysis_model(review)
    return result[0]['label']

# test_function_code --------------------

def test_sentiment_analysis():
    '''
    This function tests the sentiment_analysis function with some test cases.
    '''
    positive_review = 'I love this product!'
    negative_review = 'I hate this product!'
    neutral_review = 'This product is okay.'
    assert sentiment_analysis(positive_review) == 'POSITIVE'
    assert sentiment_analysis(negative_review) == 'NEGATIVE'
    try:
        sentiment_analysis(neutral_review)
    except ValueError:
        pass
    else:
        raise AssertionError('Expected a ValueError.')
    print('All Tests Passed')

# call_test_function_code --------------------

test_sentiment_analysis()