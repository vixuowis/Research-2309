# requirements_file --------------------

!pip install -U torch transformers

# function_import --------------------

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# function_code --------------------

def classify_review_sentiment(review_text):
    """
    Classify the sentiment of a customer review as positive or negative.

    Parameters:
    review_text (str): The text of the review to classify.

    Returns:
    str: 'positive' or 'negative' indicating the sentiment of the review.
    """
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    inputs = tokenizer(review_text, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    sentiment = model.config.id2label[predicted_class_id]
    return sentiment

# test_function_code --------------------

def test_classify_review_sentiment():
    print("Testing started.")
    # Testing with positive sentiment review
    positive_review = 'This is the best book I have ever read!'
    assert classify_review_sentiment(positive_review) == 'positive', f"Test case failed: {positive_review}"

    # Testing with negative sentiment review
    negative_review = 'This is the worst book ever. Totally disappointed.'
    assert classify_review_sentiment(negative_review) == 'negative', f"Test case failed: {negative_review}"

    # Testing with neutral sentiment review
    neutral_review = 'The book is okay, nothing special.'
    sentiment = classify_review_sentiment(neutral_review)
    assert sentiment in ['positive', 'negative'], f"Test case failed: {neutral_review}"
    print("Testing finished.")