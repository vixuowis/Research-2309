# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_customer_sentiment(customer_message: str) -> dict:
    """
    Analyze the sentiment of a customer message using a pre-trained BERT-based model.

    Args:
        customer_message (str): The customer's message to be analyzed.

    Returns:
        dict: The result of the sentiment analysis. The keys are 'label' and 'score'.
            'label' is the sentiment label (POS, NEG, NEU).
            'score' is the confidence score of the label.
    """
    sentiment_analyzer = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    result = sentiment_analyzer(customer_message)
    return result[0]

# test_function_code --------------------

def test_analyze_customer_sentiment():
    """
    Test the function analyze_customer_sentiment.
    """
    test_message = 'El servicio es excelente, estoy muy satisfecho con mi compañía de telecomunicaciones.'
    result = analyze_customer_sentiment(test_message)
    assert isinstance(result, dict)
    assert 'label' in result
    assert 'score' in result
    assert result['label'] in ['POS', 'NEG', 'NEU']

# call_test_function_code --------------------

test_analyze_customer_sentiment()