from transformers import pipeline


def classify_emotion(user_message: str) -> str:
    """
    Classify the emotion of a user's message using a pre-trained model.

    Args:
        user_message (str): The user's message to be classified.

    Returns:
        str: The classified emotion of the user's message.
    """
    emotion_classifier = pipeline('sentiment-analysis', model='michellejieli/emotion_text_classifier')
    emotion_result = emotion_classifier(user_message)
    return emotion_result[0]['label']