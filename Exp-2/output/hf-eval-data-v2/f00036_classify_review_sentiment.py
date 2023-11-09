# function_import --------------------

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# function_code --------------------

def classify_review_sentiment(review):
    """
    Classify the sentiment of a review using the DistilBertForSequenceClassification model.

    Args:
        review (str): The review text to be classified.

    Returns:
        str: The predicted sentiment of the review ('POSITIVE' or 'NEGATIVE').
    """
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    inputs = tokenizer(review, return_tensors='pt')
    logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    sentiment = model.config.id2label[predicted_class_id]
    return sentiment

# test_function_code --------------------

def test_classify_review_sentiment():
    """
    Test the classify_review_sentiment function with some example reviews.
    """
    positive_review = 'I really enjoyed this book!'
    negative_review = 'I did not like this book at all.'
    assert classify_review_sentiment(positive_review) == 'POSITIVE'
    assert classify_review_sentiment(negative_review) == 'NEGATIVE'

# call_test_function_code --------------------

test_classify_review_sentiment()