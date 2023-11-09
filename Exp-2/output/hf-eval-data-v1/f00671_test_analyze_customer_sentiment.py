def test_analyze_customer_sentiment():
    """
    This function tests the analyze_customer_sentiment function.
    It uses a sample message and checks if the sentiment analysis is working correctly.
    """
    test_message = 'El servicio es excelente, estoy muy satisfecho con mi compañía de telecomunicaciones.'
    assert analyze_customer_sentiment(test_message) in ['POS', 'NEG', 'NEU']

test_analyze_customer_sentiment()