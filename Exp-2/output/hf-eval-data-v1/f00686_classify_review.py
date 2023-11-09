from transformers import pipeline

def classify_review(review: str, categories: list):
    """
    Classify a review into one of the given categories using a zero-shot classification model.

    Parameters:
    review (str): The review to classify.
    categories (list): The list of categories to classify the review into.

    Returns:
    dict: The classification results.
    """
    classifier = pipeline('zero-shot-classification', model='vicgalle/xlm-roberta-large-xnli-anli')
    result = classifier(review, categories)
    return result