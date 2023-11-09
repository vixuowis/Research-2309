# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_sentiment(text):
    """
    Classify the sentiment of a given text using zero-shot classification.

    Args:
        text (str): The text to be classified.

    Returns:
        dict: A dictionary containing the classification results.
    """
    nlp = pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-6')
    result = nlp(text, ['positive', 'negative'])
    return result

# test_function_code --------------------

def test_classify_sentiment():
    """
    Test the classify_sentiment function.
    """
    test_text = 'The website text about technology'
    result = classify_sentiment(test_text)
    assert isinstance(result, dict)
    assert 'labels' in result
    assert 'scores' in result
    assert result['labels'][0] in ['positive', 'negative']

# call_test_function_code --------------------

test_classify_sentiment()