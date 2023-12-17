# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(review):
    """
    Analyze the sentiment of a given review using Hugging Face Transformers.

    Parameters:
    review (str): The review text to be analyzed.

    Returns:
    dict: A dictionary containing the label and score of the sentiment analysis.
    """
    sentiment_analysis = pipeline('text-classification', model='Seethal/sentiment_analysis_generic_dataset')
    result = sentiment_analysis(review)
    return result[0]

# test_function_code --------------------

def test_analyze_sentiment():
    print("Testing analyze_sentiment function")

    # Test case 1: Positive sentiment
    review_positive = "I absolutely love this product! Outstanding quality."
    result_positive = analyze_sentiment(review_positive)
    assert result_positive['label'] == 'POSITIVE', f"Test case 1 failed: {result_positive}"

    # Test case 2: Negative sentiment
    review_negative = "This is the worst product I have ever bought. Totally disappointed."
    result_negative = analyze_sentiment(review_negative)
    assert result_negative['label'] == 'NEGATIVE', f"Test case 2 failed: {result_negative}"

    print("All test cases passed.")

# Run the test function
test_analyze_sentiment()