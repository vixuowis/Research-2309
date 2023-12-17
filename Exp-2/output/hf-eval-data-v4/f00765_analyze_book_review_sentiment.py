# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_book_review_sentiment(text):
    '''
    Analyze the sentiment of a book review summary.

    Parameters:
    - text (str): A summary of the book review to be analyzed.

    Returns:
    - dict: The result of the sentiment analysis containing label and score.
    '''
    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    return sentiment_pipeline(text)

# test_function_code --------------------

def test_analyze_book_review_sentiment():
    print("Testing started.")

    # Test case with positive sentiment
    summary_positive = "The book is well-written, engaging, and insightful."
    assert analyze_book_review_sentiment(summary_positive)[0]['label'].startswith('5'), "Test case with positive sentiment failed."

    # Test case with negative sentiment
    summary_negative = "The book was dull and uninteresting."
    assert analyze_book_review_sentiment(summary_negative)[0]['label'].startswith('1'), "Test case with negative sentiment failed."

    print("All test cases passed.")

# Run the test function
test_analyze_book_review_sentiment()