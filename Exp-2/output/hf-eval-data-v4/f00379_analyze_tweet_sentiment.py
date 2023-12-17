# requirements_file --------------------

!pip install -U transformers numpy scipy

# function_import --------------------

from transformers import pipeline, AutoModel, AutoTokenizer

# function_code --------------------

def analyze_tweet_sentiment(tweet_text):
    model_path = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    sentiment_task = pipeline('sentiment-analysis', model=AutoModel.from_pretrained(model_path), tokenizer=AutoTokenizer.from_pretrained(model_path))
    sentiment_result = sentiment_task(tweet_text)
    return sentiment_result

# test_function_code --------------------

def test_analyze_tweet_sentiment():
    print("Testing started.")
    positive_tweet = "I love the new product from XYZ Corp!"
    neutral_tweet = "Just saw an advert for XYZ Corp's new product."
    negative_tweet = "Really disappointed with the new product from XYZ Corp."

    print("Testing positive sentiment tweet.")
    assert analyze_tweet_sentiment(positive_tweet)[0]['label'] == 'POSITIVE', "Positive sentiment test failed."

    print("Testing neutral sentiment tweet.")
    assert analyze_tweet_sentiment(neutral_tweet)[0]['label'] == 'NEUTRAL', "Neutral sentiment test failed."

    print("Testing negative sentiment tweet.")
    assert analyze_tweet_sentiment(negative_tweet)[0]['label'] == 'NEGATIVE', "Negative sentiment test failed."
    print("Testing finished.")

test_analyze_tweet_sentiment()