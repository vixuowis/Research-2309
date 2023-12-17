# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_customer_sentiment(message: str) -> dict:
    """Analyze the sentiment of a customer support message.

    Args:
        message (str): A text message from a customer.

    Returns:
        dict: A dictionary containing the sentiment analysis result with keys 'label' and 'score'.

    Raises:
        ValueError: If the input message is empty.
    """
    if not message:
        raise ValueError('Input message is empty.')
    sentiment_analyzer = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    result = sentiment_analyzer(message)
    return result[0]

# test_function_code --------------------

def test_analyze_customer_sentiment():
    print("Testing started.")

    # Test case 1: Positive message
    print("Testing case [1/3] started.")
    message = "El servicio es excelente, estoy muy satisfecho con mi compañía de telecomunicaciones."
    result = analyze_customer_sentiment(message)
    assert result['label'] == 'POS', f"Test case [1/3] failed: {result}"

    # Test case 2: Negative message
    print("Testing case [2/3] started.")
    message = "Estoy muy descontento con el servicio, nunca funciona correctamente."
    result = analyze_customer_sentiment(message)
    assert result['label'] == 'NEG', f"Test case [2/3] failed: {result}"

    # Test case 3: Empty message
    print("Testing case [3/3] started.")
    message = ""
    try:
        result = analyze_customer_sentiment(message)
        assert False, "Test case [3/3] failed: No ValueError raised for empty message"
    except ValueError as e:
        assert str(e) == 'Input message is empty.', f"Test case [3/3] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_customer_sentiment()