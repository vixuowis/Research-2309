# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(text):
    """Analyze the sentiment of a given text using a pre-trained model.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: The sentiment analysis result which includes label and score.

    Raises:
        ValueError: If the input text is not a string or is empty.
    """
    # Validate the input text
    if not isinstance(text, str) or not text:
        raise ValueError('The input text must be a non-empty string.')

    # Initialize the sentiment analysis model
    sentiment_analysis = pipeline('sentiment-analysis', model='siebert/sentiment-roberta-large-english')
    # Analyze the sentiment
    result = sentiment_analysis(text)
    return result

# test_function_code --------------------

def test_analyze_sentiment():
    print("Testing started.")
    # Test case 1: Positive sentiment
    print("Testing case [1/3] started.")
    result = analyze_sentiment("I love the new product!")
    assert result[0]['label'] == 'POSITIVE', f"Test case [1/3] failed: Expected POSITIVE, got {result[0]['label']}"

    # Test case 2: Negative sentiment
    print("Testing case [2/3] started.")
    result = analyze_sentiment("I hate this product.")
    assert result[0]['label'] == 'NEGATIVE', f"Test case [2/3] failed: Expected NEGATIVE, got {result[0]['label']}"

    # Test case 3: Raise ValueError on empty input
    print("Testing case [3/3] started.")
    try:
        analyze_sentiment("")
        assert False, "Test case [3/3] failed: ValueError was not raised on empty input."
    except ValueError as e:
        assert str(e) == 'The input text must be a non-empty string.', f"Test case [3/3] failed: {str(e)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_sentiment()