from transformers import pipeline


def classify_emotion(text):
    """
    Classify the emotion of a given text using the 'joeddav/distilbert-base-uncased-go-emotions-student' model.

    Args:
        text (str): The text to be classified.

    Returns:
        dict: The classified emotion of the text.
    """
    nlp = pipeline('text-classification', model='joeddav/distilbert-base-uncased-go-emotions-student')
    result = nlp(text)
    return result