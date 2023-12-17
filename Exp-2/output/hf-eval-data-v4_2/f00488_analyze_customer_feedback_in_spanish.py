# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_customer_feedback_in_spanish(customer_feedback_in_spanish):
    """Analyze the sentiment of a given customer feedback in Spanish.

    Args:
        customer_feedback_in_spanish (str): The text of the customer feedback in Spanish to analyze.

    Returns:
        dict: A dictionary result of sentiment analysis with label and score.

    Raises:
        ValueError: If `customer_feedback_in_spanish` is not a string or is empty.
    """
    if not isinstance(customer_feedback_in_spanish, str) or not customer_feedback_in_spanish:
        raise ValueError('The input must be a non-empty string.')
    feedback_sentiment = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    sentiment_result = feedback_sentiment(customer_feedback_in_spanish)
    return sentiment_result

# test_function_code --------------------

def test_analyze_customer_feedback_in_spanish():
    print("Testing started.")
    # Simulate customer feedback in Spanish
    feedback_examples = [
        ('El producto funciona muy bien!', 'POS'),
        ('No estoy satisfecho con la compra.', 'NEG'),
        ('Es un producto más o menos, nada especial.', 'NEU')
    ]

    for i, (feedback, expected_label) in enumerate(feedback_examples, start=1):
        print(f"Testing case [{i}/{len(feedback_examples)}] started.")
        sentiment_result = analyze_customer_feedback_in_spanish(feedback)
        assert sentiment_result[0]['label'] == expected_label, f"Test case [{i}/{len(feedback_examples)}] failed: Expected {expected_label}, got {sentiment_result[0]['label']}"
        print(f"Test case [{i}/{len(feedback_examples)}] succeeded.")
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_customer_feedback_in_spanish()