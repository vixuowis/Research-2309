# requirements_file --------------------

import subprocess

requirements = ["torch", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# function_code --------------------

def classify_sentiment(text):
    """
    Classify the sentiment of the given text as positive or negative.

    Args:
        text (str): The customer review text to be classified.

    Returns:
        str: The sentiment classification result, either 'positive' or 'negative'.

    Raises:
        ValueError: If the `text` is empty.
    """
    if not text:
        raise ValueError('Input text must not be empty.')

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    sentiment = model.config.id2label[predicted_class_id]
    return sentiment

# test_function_code --------------------

def test_classify_sentiment():
    print("Testing started.")

    # Test case 1: Positive sentiment
    print("Testing case [1/3] started.")
    assert classify_sentiment('I really enjoyed this book!') == 'positive', "Test case [1/3] failed: Expected 'positive' sentiment"

    # Test case 2: Negative sentiment
    print("Testing case [2/3] started.")
    assert classify_sentiment('This is the worst book I have ever read.') == 'negative', "Test case [2/3] failed: Expected 'negative' sentiment"

    # Test case 3: Raise ValueError when input text is empty
    print("Testing case [3/3] started.")
    try:
        classify_sentiment('')
        assert False, 'Test case [3/3] failed: ValueError not raised for empty text.'
    except ValueError as e:
        assert str(e) == 'Input text must not be empty.', f'Test case [3/3] failed: {str(e)}'
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_sentiment()