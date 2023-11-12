# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# function_code --------------------

def analyze_review_sentiment(review_text):
    """
    Analyze the sentiment of a restaurant review using a pre-trained model.

    Args:
        review_text (str): The text of the restaurant review.

    Returns:
        str: The sentiment of the review ('positive' or 'negative').

    Raises:
        OSError: If there is an error loading the pre-trained model or tokenizing the input text.
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
    Test the analyze_review_sentiment function.
    """
    positive_review = 'The food was delicious and the service was excellent.'
    negative_review = 'The food was terrible and the service was poor.'
    assert analyze_review_sentiment(positive_review) == 'positive'
    assert analyze_review_sentiment(negative_review) == 'negative'
    return 'All Tests Passed'

# call_test_function_code --------------------

if __name__ == '__main__':
    print(test_analyze_review_sentiment())