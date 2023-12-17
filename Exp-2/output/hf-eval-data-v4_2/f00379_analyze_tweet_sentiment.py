# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline, AutoModel, AutoTokenizer

# function_code --------------------

def analyze_tweet_sentiment(tweet: str) -> dict:
    """
    Analyze the sentiment of a given tweet.

    Args:
        tweet (str): The tweet text to analyze.

    Returns:
        dict: The sentiment analysis result containing label and score.

    Raises:
        ValueError: If the tweet text is empty.
    """
    if not tweet:
        raise ValueError('The tweet text cannot be empty.')
    model_path = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    sentiment_task = pipeline('sentiment-analysis', model=AutoModel.from_pretrained(model_path), tokenizer=AutoTokenizer.from_pretrained(model_path))
    return sentiment_task(tweet)

# test_function_code --------------------

def test_analyze_tweet_sentiment():
    print("Testing started.")

    # Test case 1: Positive sentiment
    tweet_positive = "I love the new product!"
    print("Testing case [1/3] started.")
    sentiment_positive = analyze_tweet_sentiment(tweet_positive)
    assert sentiment_positive[0]['label'] in ['POSITIVE'], f"Test case [1/3] failed: {sentiment_positive}"

    # Test case 2: Negative sentiment
    tweet_negative = "I hate this new update."
    print("Testing case [2/3] started.")
    sentiment_negative = analyze_tweet_sentiment(tweet_negative)
    assert sentiment_negative[0]['label'] in ['NEGATIVE'], f"Test case [2/3] failed: {sentiment_negative}"

    # Test case 3: Neutral sentiment
    tweet_neutral = "It's an average product."
    print("Testing case [3/3] started.")
    sentiment_neutral = analyze_tweet_sentiment(tweet_neutral)
    assert sentiment_neutral[0]['label'] in ['NEUTRAL'], f"Test case [3/3] failed: {sentiment_neutral}"
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_tweet_sentiment()