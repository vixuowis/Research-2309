# function_import --------------------

from transformers import pipeline

# function_code --------------------

def sentiment_analysis(text):
    """
    Analyze the sentiment of a given text using a pre-trained model from the transformers library.

    Args:
        text (str): The text to be analyzed.

    Returns:
        dict: A dictionary containing the sentiment ('label') and the confidence score ('score').
    """
    sentiment_analysis_model = pipeline('sentiment-analysis', model='siebert/sentiment-roberta-large-english')
    result = sentiment_analysis_model(text)
    return result[0]

# test_function_code --------------------

def test_sentiment_analysis():
    """
    Test the sentiment_analysis function with some test cases.
    """
    assert sentiment_analysis('I love the new product!')['label'] == 'POSITIVE'
    assert sentiment_analysis('I hate the new product!')['label'] == 'NEGATIVE'
    assert sentiment_analysis('The new product is okay.')['label'] == 'POSITIVE'
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_sentiment_analysis())