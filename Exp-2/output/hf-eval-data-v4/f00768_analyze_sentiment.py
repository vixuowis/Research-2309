# requirements_file --------------------

!pip install -U transformers==4.18.0 torch==1.10.0+cu111

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# function_code --------------------

def analyze_sentiment(review_text):
    '''
    Analyze the sentiment of a restaurant review using Hugging Face's transformers.

    :param review_text: str, The review text to analyze.
    :return: str, 'positive' or 'negative' indicating the sentiment of the review.
    '''
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('potatobunny/results-yelp')

    # Tokenize the input text and get classification
    inputs = tokenizer(review_text, truncation=True, padding=True, return_tensors='pt')
    outputs = model(**inputs)
    label_id = outputs.logits.argmax(-1).item()

    # Convert to human-readable label
    return 'positive' if label_id == 1 else 'negative'

# test_function_code --------------------

def test_analyze_sentiment():
    print("Testing analyze_sentiment function.")

    # Test case 1: Positive review
    positive_review = "The food was excellent, service was great!"
    assert analyze_sentiment(positive_review) == 'positive', "Test case 1 failed: Expected 'positive'"

    # Test case 2: Negative review
    negative_review = "The food was terrible, wouldn't recommend."
    assert analyze_sentiment(negative_review) == 'negative', "Test case 2 failed: Expected 'negative'"

    # Test case 3: Neutral review treated as negative
    neutral_review = "The food was okay, nothing special."
    assert analyze_sentiment(neutral_review) == 'negative', "Test case 3 failed: Expected 'negative'"

    print("All test cases passed!")
