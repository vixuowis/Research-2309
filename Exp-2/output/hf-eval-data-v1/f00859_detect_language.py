from transformers import pipeline


def detect_language(text):
    """
    Detects the language of the given text using a pre-trained model.

    Args:
        text (str): The text whose language is to be detected.

    Returns:
        dict: A dictionary containing the detected language and its confidence score.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')

    # Create a text classification pipeline using the pre-trained model
    language_detection = pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection')

    # Detect the language of the input text
    result = language_detection(text)

    return result