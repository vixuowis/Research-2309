# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_book_review_sentiment(summary: str) -> str:
    """
    Analyze sentiment of a book review summary using a pre-trained sentiment analysis model.

    Args:
        summary (str): A summary text of a book review.

    Returns:
        str: The sentiment analysis result as a label (e.g., 'POSITIVE' or 'NEGATIVE').

    Raises:
        ValueError: If the summary is empty.
    """
    if not summary:
        raise ValueError('The summary cannot be empty.')
    
    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    result = sentiment_pipeline(summary)
    
    sentiment_label = result[0]['label']
    return sentiment_label

# test_function_code --------------------

from transformers import pipeline
def test_analyze_book_review_sentiment():
    print("Testing started.")
    # Assuming the sentiment_pipeline is mocked or instantiated with a specific model
    
    # Test case 1: A positive summary
    print("Testing case [1/3] started.")
    summary_positive = "The book is well-written, engaging, and insightful."
    assert analyze_book_review_sentiment(summary_positive) == 'POSITIVE', f"Test case [1/3] failed: Expected POSITIVE sentiment."

    # Test case 2: A negative summary
    print("Testing case [2/3] started.")
    summary_negative = "The book was boring and poorly written."
    assert analyze_book_review_sentiment(summary_negative) == 'NEGATIVE', f"Test case [2/3] failed: Expected NEGATIVE sentiment."

    # Test case 3: An empty summary
    print("Testing case [3/3] started.")
    summary_empty = ""
    try:
        analyze_book_review_sentiment(summary_empty)
        assert False, "Test case [3/3] failed: Expected ValueError for empty summary."
    except ValueError:
        assert True
    
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_book_review_sentiment()