# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_customer_review_sentiment(review_text):
    """
    Analyze the sentiment of a customer review.
    
    The function uses the 'nlptown/bert-base-multilingual-uncased-sentiment' model to analyze the sentiment of the given review text.
    The sentiment analysis model is capable of understanding multiple languages and outputs a star rating.

    Parameters:
    - review_text (str): The customer review text.

    Returns:
    - dict: A dictionary containing the sentiment analysis result.
    """
    # Initialize the sentiment analysis pipeline
    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    # Analyze the sentiment of the review
    result = sentiment_pipeline(review_text)
    return result

# test_function_code --------------------

def test_analyze_customer_review_sentiment():
    print("Testing analyze_customer_review_sentiment function.")
    # Positive test case
    positive_review = "¡Esto es maravilloso! Me encanta."
    positive_result = analyze_customer_review_sentiment(positive_review)
    assert positive_result[0]['label'] == '5 stars', f"Positive test case failed: {positive_result}"
    print("Positive test case passed.")

    # Negative test case
    negative_review = "No me gusta este producto."
    negative_result = analyze_customer_review_sentiment(negative_review)
    assert negative_result[0]['label'] == '1 star', f"Negative test case failed: {negative_result}"
    print("Negative test case passed.")

    # Neutral test case
    neutral_review = "Es un producto más o menos."
    neutral_result = analyze_customer_review_sentiment(neutral_review)
    assert neutral_result[0]['label'] == '3 stars', f"Neutral test case failed: {neutral_result}"
    print("Neutral test case passed.")

    print("All tests passed.")

# Run the test function
print("Starting function tests...")
test_analyze_customer_review_sentiment()