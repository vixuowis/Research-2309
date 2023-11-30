# function_import --------------------

from transformers import pipeline

# function_code --------------------

def sentiment_analysis(text: str) -> dict:
    """
    This function uses the zero-shot classification model from the transformers library to perform sentiment analysis on a given text.

    Args:
        text (str): The text to be analyzed.

    Returns:
        dict: The sentiment analysis result.
    """
    
    # Initialize the pipeline with pretrained model.
    classifier = pipeline("zero-shot-classification", device=0)
        
    # Classify text
    return classifier(text, ["positive", "negative"])


# test_function_code --------------------

def test_sentiment_analysis():
    """
    This function tests the sentiment_analysis function with different test cases.
    """
    assert sentiment_analysis('The movie was great!')['labels'][0] == 'positive'
    assert sentiment_analysis('I hate this product.')['labels'][0] == 'negative'
    assert sentiment_analysis('This is a neutral statement.')['labels'][0] in ['positive', 'negative']
    return 'All Tests Passed'


# call_test_function_code --------------------

test_sentiment_analysis()