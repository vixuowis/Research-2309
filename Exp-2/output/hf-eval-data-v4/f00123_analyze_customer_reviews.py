# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_customer_reviews(review_text):
    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    review_sentiment = sentiment_pipeline(review_text)
    return review_sentiment

# test_function_code --------------------

def test_analyze_customer_reviews():
    print("Testing analyze_customer_reviews function.")
    
    # Test case 1: Check analysis of a positive review
    positive_review = "I love this product!"
    result = analyze_customer_reviews(positive_review)
    assert 'star' in result[0]['label'], f"Test case failed: Sentiment label not found in result {result}"
    
    # Test case 2: Check analysis of a negative review
    negative_review = "This product is terrible."
    result = analyze_customer_reviews(negative_review)
    assert 'star' in result[0]['label'], f"Test case failed: Sentiment label not found in result {result}"

    print("All test cases passed.")

# Run the test function
test_analyze_customer_reviews()