# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_customer_sentiment(message):
    """
    Analyze the sentiment of the given customer message using a pre-trained BERT-based model.
    
    Parameters:
    - message (str): The customer's text message to be analyzed.
    
    Returns:
    - dict: A dictionary with the message's sentiment analysis result.
    """
    # Initialize the sentiment analysis pipeline
    sentiment_analyzer = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    
    # Analyze the sentiment of the message
    result = sentiment_analyzer(message)
    
    return result

# test_function_code --------------------

def test_analyze_customer_sentiment():
    print("Testing started.")
    
    # Test case 1: Positive sentiment message
    print("Testing case [1/3] started.")
    positive_message = "El servicio es excelente, estoy muy satisfecho con mi compañía de telecomunicaciones."
    positive_result = analyze_customer_sentiment(positive_message)
    assert positive_result[0]['label'] == 'POS' and positive_result[0]['score'] > 0.5, f"Test case [1/3] failed: {positive_result}"
    
    # Test case 2: Negative sentiment message
    print("Testing case [2/3] started.")
    negative_message = "Estoy decepcionado con la poca cobertura que ofrece la empresa."
    negative_result = analyze_customer_sentiment(negative_message)
    assert negative_result[0]['label'] == 'NEG' and negative_result[0]['score'] > 0.5, f"Test case [2/3] failed: {negative_result}"

    # Test case 3: Neutral sentiment message
    print("Testing case [3/3] started.")
    neutral_message = "Recibí la factura de este mes."
    neutral_result = analyze_customer_sentiment(neutral_message)
    assert neutral_result[0]['label'] == 'NEU' and neutral_result[0]['score'] > 0.5, f"Test case [3/3] failed: {neutral_result}"
    
    print("Testing finished.")

# Run the test function
test_analyze_customer_sentiment()