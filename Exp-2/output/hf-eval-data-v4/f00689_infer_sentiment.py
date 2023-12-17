# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def infer_sentiment(comment):
    """
    Infer the sentiment of a given user comment using the zero-shot classification model.

    Parameters:
    comment (str): The user comment to analyze.

    Returns:
    dict: A dictionary containing the sentiment and confidence score.
    """
    model_name = 'valhalla/distilbart-mnli-12-6'
    classifier = pipeline('zero-shot-classification', model=model_name)
    labels = ['positive', 'negative']
    result = classifier(comment, labels)
    sentiment = result['labels'][0]
    confidence = result['scores'][0]
    return {'sentiment': sentiment, 'confidence': confidence}

# test_function_code --------------------

def test_infer_sentiment():
    print("Testing infer_sentiment function.")
    comment = "I love this new update, everything works flawlessly!"
    result = infer_sentiment(comment)

    assert result['sentiment'] == 'positive', f"Expected sentiment to be 'positive', but got '{result['sentiment']}' instead."
    assert result['confidence'] >= 0.5, f"Expected confidence to be at least 0.5, but got {result['confidence']} instead."
    print("All tests passed.")