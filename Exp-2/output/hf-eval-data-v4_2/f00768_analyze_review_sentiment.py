# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# function_code --------------------

def analyze_review_sentiment(review_text: str) -> str:
    """
    Analyze the sentiment of a restaurant review using the Hugging Face Transformers model.

    Args:
        review_text (str): The text of the restaurant review to analyze.

    Returns:
        str: The sentiment analysis result, either 'positive' or 'negative'.

    Raises:
        ValueError: If the review_text is an empty string.
    """
    if not review_text:
        raise ValueError('The review text cannot be empty.')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('potatobunny/results-yelp')
    inputs = tokenizer(review_text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)

    # Convert the model output to positive or negative sentiment
    sentiment_label = 'positive' if outputs.logits.argmax(-1).item() == 1 else 'negative'
    return sentiment_label

# test_function_code --------------------

def test_analyze_review_sentiment():
    print("Testing started.")

    # Test case 1: Positive sentiment
    print("Testing case [1/2] started.")
    positive_review = 'This restaurant was absolutely wonderful! Delicious food and great service.'
    assert analyze_review_sentiment(positive_review) == 'positive', f"Test case [1/2] failed: {positive_review}"

    # Test case 2: Negative sentiment
    print("Testing case [2/2] started.")
    negative_review = 'Terrible experience, would not recommend. Overcooked food and rude staff.'
    assert analyze_review_sentiment(negative_review) == 'negative', f"Test case [2/2] failed: {negative_review}"
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_review_sentiment()