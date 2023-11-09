from transformers import pipeline


def detect_emotion(user_response_text):
    """
    Detects the emotion from the user's response text using a sentiment analysis pipeline.

    Args:
        user_response_text (str): The user's response text.

    Returns:
        dict: The predicted emotion and its score.
    """
    emotion_detector = pipeline('sentiment-analysis', model='michellejieli/emotion_text_classifier')
    user_emotion = emotion_detector(user_response_text)
    return user_emotion