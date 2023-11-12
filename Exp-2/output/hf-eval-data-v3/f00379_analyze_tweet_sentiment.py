# function_import --------------------

from transformers import pipeline, AutoModel, AutoTokenizer

# function_code --------------------

def analyze_tweet_sentiment(tweet_text):
    """
    Analyze the sentiment of a given tweet using a pre-trained model.

    Args:
        tweet_text (str): The text of the tweet to analyze.

    Returns:
        str: The sentiment of the tweet ('POSITIVE', 'NEGATIVE', or 'NEUTRAL').
    """
    model_path = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    sentiment_task = pipeline('sentiment-analysis', model=AutoModel.from_pretrained(model_path), tokenizer=AutoTokenizer.from_pretrained(model_path))
    sentiment_result = sentiment_task(tweet_text)
    return sentiment_result[0]['label']

# test_function_code --------------------

def test_analyze_tweet_sentiment():
    """
    Test the analyze_tweet_sentiment function with some example tweets.
    """
    assert analyze_tweet_sentiment('I love the new product') == 'POSITIVE'
    assert analyze_tweet_sentiment('I hate the new product') == 'NEGATIVE'
    assert analyze_tweet_sentiment('The new product is okay') == 'NEUTRAL'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_analyze_tweet_sentiment()