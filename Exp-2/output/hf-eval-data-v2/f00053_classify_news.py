# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_news(text):
    """
    Classify a piece of news into categories: technology, sports, or politics.

    Args:
        text (str): The news text to be classified.

    Returns:
        dict: A dictionary containing the labels and their corresponding scores.
    """
    candidate_labels = ["technology", "sports", "politics"]
    classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-roberta-base")
    result = classifier(text, candidate_labels)
    return result

# test_function_code --------------------

def test_classify_news():
    """
    Test the classify_news function.
    """
    test_text = "Apple just announced the newest iPhone X"
    result = classify_news(test_text)
    assert isinstance(result, dict)
    assert 'labels' in result and 'scores' in result
    assert set(result['labels']) == set(["technology", "sports", "politics"])
    assert all(isinstance(score, float) for score in result['scores'])

# call_test_function_code --------------------

test_classify_news()