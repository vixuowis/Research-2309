# function_import --------------------

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# function_code --------------------

def analyze_review_sentiment(review_text):
    """
    Analyze the sentiment of a restaurant review using a fine-tuned BERT model.

    Args:
        review_text (str): The text of the restaurant review to analyze.

    Returns:
        str: The sentiment of the review ('positive' or 'negative').
    """
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    config = AutoConfig.from_pretrained('potatobunny/results-yelp')
    model = AutoModelForSequenceClassification.from_pretrained('potatobunny/results-yelp', config=config)
    inputs = tokenizer(review_text, return_tensors='pt')
    outputs = model(**inputs)
    sentiment = 'positive' if outputs.logits[0, 1] > outputs.logits[0, 0] else 'negative'
    return sentiment

# test_function_code --------------------

def test_analyze_review_sentiment():
    """
    Test the analyze_review_sentiment function with some example reviews.
    """
    positive_review = 'The food was delicious and the service was excellent.'
    negative_review = 'The food was terrible and the service was poor.'
    assert analyze_review_sentiment(positive_review) == 'positive'
    assert analyze_review_sentiment(negative_review) == 'negative'

# call_test_function_code --------------------

test_analyze_review_sentiment()