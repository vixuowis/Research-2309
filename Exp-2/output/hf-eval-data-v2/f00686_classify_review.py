# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_review(review_text, categories):
    """
    Classify a given review text into one of the provided categories using a zero-shot classification model.

    Args:
        review_text (str): The text of the review to be classified.
        categories (list): A list of categories to classify the review into.

    Returns:
        dict: A dictionary containing the classification results.
    """
    classifier = pipeline('zero-shot-classification', model='vicgalle/xlm-roberta-large-xnli-anli')
    result = classifier(review_text, categories)
    return result

# test_function_code --------------------

def test_classify_review():
    """
    Test the classify_review function with a sample review and categories.
    """
    review_text = 'Algún día iré a ver el mundo'
    categories = ['viaje', 'cocina', 'danza']
    result = classify_review(review_text, categories)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'labels' in result, 'The result dictionary should contain a labels key.'
    assert 'scores' in result, 'The result dictionary should contain a scores key.'

# call_test_function_code --------------------

test_classify_review()