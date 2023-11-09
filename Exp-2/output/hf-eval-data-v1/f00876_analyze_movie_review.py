from transformers import pipeline


def analyze_movie_review(review_text: str) -> dict:
    """
    Analyze a movie review using zero-shot classification.

    This function uses the 'valhalla/distilbart-mnli-12-6' model from the Transformers library
    to perform zero-shot classification on a given movie review. The review is classified
    into one of two categories: 'positive' or 'negative'.

    Args:
        review_text (str): The text of the movie review to analyze.

    Returns:
        dict: A dictionary containing the classification results.
    """
    nlp = pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-6')
    result = nlp(review_text, ['positive', 'negative'])
    return result