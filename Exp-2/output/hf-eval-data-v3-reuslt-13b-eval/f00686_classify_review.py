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
    # Load the zero-shot classification pipeline.
    try:
        pipe = pipeline("zero-shot-classification")
    except OSError as error:
        raise OSError(f"There was a problem loading the zero-shot classification model - check that it is installed correctly.") from error

    # Classify the review using the selected categories.
    try:
        class_results = pipe(review_text, candidate_labels=categories)
    except OSError as error:
        raise OSError("There was a problem classifying the review text - check that it is not blank.") from error

    # Create a results dictionary containing only the fields required for this function.
    return {
        "text": review_text,
        "categories": categories,
        "scores": dict(zip(class_results["labels"], class_results["scores"]))
    }


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