# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(review):
    """
    Analyze the sentiment of a given review using a pre-trained model.

    Args:
        review (str): The review to be analyzed.

    Returns:
        dict: The result of the sentiment analysis, including the label ('POSITIVE' or 'NEGATIVE') and the score.
    """
    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    result = sentiment_pipeline(review)
    return result

# test_function_code --------------------

def test_analyze_sentiment():
    """
    Test the analyze_sentiment function with some test cases.
    """
    assert analyze_sentiment('¡Esto es maravilloso! Me encanta.')[0]['label'] == 'POSITIVE'
    assert analyze_sentiment('No me gusta este producto.')[0]['label'] == 'NEGATIVE'
    assert analyze_sentiment('Este producto es increíble.')[0]['label'] == 'POSITIVE'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_analyze_sentiment()