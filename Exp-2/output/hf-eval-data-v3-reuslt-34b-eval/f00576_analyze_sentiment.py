# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(review_text):
    """
    Analyze the sentiment of a given text using the 'finiteautomata/beto-sentiment-analysis' model.

    Args:
        review_text (str): The text to be analyzed.

    Returns:
        dict: The sentiment analysis result. The keys are 'label' and 'score'.
    """

    nlp = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')   # Load the model

    result = nlp(review_text)[0]                             # Get the sentiment analysis of the text

    return {'label':result['label'], 'score':round(100*result['score'])}     # Return the sentiment analysis


# test_function_code --------------------

def test_analyze_sentiment():
    """
    Test the function analyze_sentiment.
    """
    assert analyze_sentiment('Me encanta este producto.')[0]['label'] == 'POS'
    assert analyze_sentiment('No me gusta este producto.')[0]['label'] == 'NEG'
    assert analyze_sentiment('Este producto es normal.')[0]['label'] == 'NEU'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_analyze_sentiment()