# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(text: str) -> dict:
    '''
    Analyze the sentiment of a given text using a pre-trained BERT-based model for sentiment analysis on Spanish texts.

    Args:
        text (str): The text to be analyzed.

    Returns:
        dict: The sentiment analysis result.
    '''
    sentiment_analyzer = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    result = sentiment_analyzer(text)
    return result

# test_function_code --------------------

def test_analyze_sentiment():
    '''
    Test the analyze_sentiment function.
    '''
    assert isinstance(analyze_sentiment('El servicio es excelente, estoy muy satisfecho con mi compañía de telecomunicaciones.'), dict)
    assert isinstance(analyze_sentiment('No estoy contento con el servicio.'), dict)
    assert isinstance(analyze_sentiment('El servicio es regular, podría mejorar.'), dict)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_analyze_sentiment()