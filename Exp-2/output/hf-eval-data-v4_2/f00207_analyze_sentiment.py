# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text.

    Args:
        text (str): The user review text to analyze.

    Returns:
        dict: A dictionary with the result of the sentiment analysis.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')
    sentiment_analyzer = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    sentiment_result = sentiment_analyzer(text)
    return sentiment_result

# test_function_code --------------------

def test_analyze_sentiment():
    print("Testing started.")
    # Test case 1: Positive sentiment
    print("Testing case [1/3] started.")
    positive_text = 'Esta aplicación funciona de maravilla, ¡excelente trabajo!'
    positive_result = analyze_sentiment(positive_text)
    assert positive_result[0]['label'] == 'POSITIVE', f"Test case [1/3] failed: Expected POSITIVE, got {positive_result[0]['label']}"

    # Test case 2: Negative sentiment
    print("Testing case [2/3] started.")
    negative_text = 'Esta aplicación es terrible, nunca funciona correctamente.'
    negative_result = analyze_sentiment(negative_text)
    assert negative_result[0]['label'] == 'NEGATIVE', f"Test case [2/3] failed: Expected NEGATIVE, got {negative_result[0]['label']}"

    # Test case 3: Neutral sentiment
    print("Testing case [3/3] started.")
    neutral_text = 'La aplicación es aceptable, pero tiene espacio para mejorar.'
    neutral_result = analyze_sentiment(neutral_text)
    assert neutral_result[0]['label'] == 'NEUTRAL', f"Test case [3/3] failed: Expected NEUTRAL, got {neutral_result[0]['label']}"
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_sentiment()