# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_customer_review_sentiment(review_text):
    """
    Analyze the sentiment of a customer review.

    Parameters:
    review_text (str): The text of the customer review to analyze.

    Returns:
    dict: A dictionary with the review sentiment analysis results.
    """
    model_path = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
    sentiment_task = pipeline('sentiment-analysis', model=model_path, tokenizer=model_path)
    result = sentiment_task(review_text)
    return result[0]

# test_function_code --------------------

def test_analyze_customer_review_sentiment():
    print("Testing analyze_customer_review_sentiment function.")
    # Sample customer reviews
    sample_reviews = [
        'Great service, very satisfied!',
        'The worst experience ever, totally disappointing.',
        'It was okay, not great but not bad either.'
    ]

    # Expected sentiments are positive, negative, and neutral respectively
    expected_sentiments = ['LABEL_0', 'LABEL_2', 'LABEL_1']

    for i, review in enumerate(sample_reviews):
        result = analyze_customer_review_sentiment(review)
        assert result['label'] == expected_sentiments[i], f"Test case failed for review: {review}"
    print("All test cases passed.")

test_analyze_customer_review_sentiment()