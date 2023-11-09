from transformers import pipeline


def predict_punctuation(novel_draft_text):
    """
    Predicts the punctuation marks needed in a given text.

    Args:
        novel_draft_text (str): The text in which to predict punctuation marks.

    Returns:
        dict: A dictionary where keys are the words from the input text and values are the predicted punctuation marks.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(novel_draft_text, str):
        raise ValueError('Input text should be a string.')

    punctuation_predictor = pipeline('token-classification', model='kredor/punctuate-all')
    predicted_punctuations = punctuation_predictor(novel_draft_text)
    return predicted_punctuations