# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_review(review_text: str, categories: list) -> dict:
    """
    Classify a review into one of the given categories using a zero-shot classification model.

    Args:
        review_text (str): The text of the review to classify.
        categories (list): A list of categories to classify the review into.

    Returns:
        dict: A dictionary containing the classification results.

    Raises:
        OSError: If there is a problem loading the model or classifying the review.
    """

    try:
        # Load zero-shot classification pipeline
        pipe = pipeline('zero-shot-classification')
        
        # Classify the review
        res = pipe(review_text, categories)
    
    except OSError as e:
        raise OSError(e)

    return res

# test_function_code --------------------

def test_classify_review():
    """
    Test the classify_review function with some example reviews and categories.
    """
    review_text1 = 'Algún día iré a ver el mundo'
    categories1 = ['viaje', 'cocina', 'danza']
    result1 = classify_review(review_text1, categories1)
    assert isinstance(result1, dict), 'The result should be a dictionary.'

    review_text2 = 'Me encanta cocinar paella'
    categories2 = ['viaje', 'cocina', 'danza']
    result2 = classify_review(review_text2, categories2)
    assert isinstance(result2, dict), 'The result should be a dictionary.'

    review_text3 = 'Bailar es mi pasión'
    categories3 = ['viaje', 'cocina', 'danza']
    result3 = classify_review(review_text3, categories3)
    assert isinstance(result3, dict), 'The result should be a dictionary.'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_classify_review()