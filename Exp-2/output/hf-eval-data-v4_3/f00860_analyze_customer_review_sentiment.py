# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_customer_review_sentiment(review_text):
    """
    Analyze the sentiment of a customer review using FinBERT model.

    Args:
        review_text (str): A text string containing the customer review to be analyzed.

    Returns:
        dict: A dictionary containing the sentiment analysis results with keys 'label' and 'score'.

    Raises:
        ValueError: If 'review_text' is not a string.
    """
    if not isinstance(review_text, str):
        raise ValueError('The review text must be a string.')

    # Initialize the sentiment analysis model
    classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')
    
    # Analyze the sentiment of the review
    result = classifier(review_text)
    return result[0]

# test_function_code --------------------

def test_analyze_customer_review_sentiment():
    print("Testing started.")

    # Test case 1: Positive sentiment
    print("Testing case [1/3] started.")
    positive_review = 'I love this financial service. It is reliable and user-friendly.'
    positive_result = analyze_customer_review_sentiment(positive_review)
    assert positive_result['label'] in ['LABEL_1', 'LABEL_2'], f"Test case [1/3] failed: {positive_result}"

    # Test case 2: Negative sentiment
    print("Testing case [2/3] started.")
    negative_review = 'This app is horrible. It always crashes and has many bugs.'
    negative_result = analyze_customer_review_sentiment(negative_review)
    assert negative_result['label'] == 'LABEL_0', f"Test case [2/3] failed: {negative_result}"

    # Test case 3: Neutral sentiment
    print("Testing case [3/3] started.")
    neutral_review = 'The app is okay, but it lacks some key features.'
    neutral_result = analyze_customer_review_sentiment(neutral_review)
    assert neutral_result['label'] == 'LABEL_1', f"Test case [3/3] failed: {neutral_result}"

    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_customer_review_sentiment()