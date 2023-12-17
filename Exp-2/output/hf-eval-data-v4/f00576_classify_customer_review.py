# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_customer_review(review_text):
    # Load the sentiment analysis model
    sentiment_model = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    # Get the sentiment result for a given review text
    sentiment_result = sentiment_model(review_text)
    # Return the sentiment analysis result
    return sentiment_result

# test_function_code --------------------

def test_classify_customer_review():
    print("Testing classify_customer_review function.")

    # Define sample customer reviews
    sample_reviews = [
        'Estoy muy satisfecho con el producto.',  # Expected POS
        'El producto no cumplió mis expectativas.',  # Expected NEG
        'El producto es justo lo que esperaba.',  # Expected NEU
    ]

    # Expected results for the sample reviews
    expected_results = ['POS', 'NEG', 'NEU']

    for i, review in enumerate(sample_reviews):
        print(f"Testing case [{i+1}/{len(sample_reviews)}] started.")
        result = classify_customer_review(review)
        assert result[0]['label'] == expected_results[i], f"Test case [{i+1}/{len(sample_reviews)}] failed: Expected {expected_results[i]}, got {result[0]['label']}"
        print(f"Testing case [{i+1}/{len(sample_reviews)}] succeeded.")

    print("Testing finished.")