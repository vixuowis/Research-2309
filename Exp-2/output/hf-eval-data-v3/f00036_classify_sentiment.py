# function_import --------------------

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# function_code --------------------

def classify_sentiment(text):
    """
    Classify the sentiment of a given text using DistilBertForSequenceClassification model.

    Args:
        text (str): The text to be classified.

    Returns:
        str: The sentiment of the text, either 'positive' or 'negative'.
    """
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    inputs = tokenizer(text, return_tensors='pt')
    logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    sentiment = model.config.id2label[predicted_class_id]
    return sentiment

# test_function_code --------------------

def test_classify_sentiment():
    assert classify_sentiment('I really enjoyed this book!') == 'positive'
    assert classify_sentiment('This is the worst book I have ever read.') == 'negative'
    assert classify_sentiment('The book is okay, not great but not bad either.') in ['positive', 'negative']
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_sentiment()