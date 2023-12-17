# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(text):
    """
    Analyze the sentiment of the provided text using a multilingual sentiment analysis model.

    Args:
        text (str): The text to be analyzed.

    Returns:
        dict: The result from the sentiment analysis containing the predicted sentiment and score.

    Raises:
        ValueError: If the text is empty.

    """
    if not text:
        raise ValueError('The text cannot be empty.')

    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    result = sentiment_pipeline(text)
    return result

# test_function_code --------------------

def test_analyze_sentiment():
    print("Testing started.")
    
    # Test case 1: Positive sentiment
    print("Testing case [1/2] started.")
    result = analyze_sentiment("¡Esto es maravilloso! Me encanta.")
    assert isinstance(result, list) and len(result) > 0, f"Test case [1/2] failed: {result}"

    # Test case 2: Empty string raises ValueError
    print("Testing case [2/2] started.")
    try:
        analyze_sentiment("")
        assert False, "Test case [2/2] failed: ValueError not raised for empty text"
    except ValueError as e:
        assert str(e) == 'The text cannot be empty.', f"Test case [2/2] failed: Incorrect error message {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_sentiment()