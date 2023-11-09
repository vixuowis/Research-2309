# function_import --------------------

from transformers import pipeline

# function_code --------------------

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

# test_function_code --------------------

def test_detect_emotion():
    """
    Tests the detect_emotion function with some sample texts.
    """
    sample_texts = ['I love this!', 'I am so angry right now.', 'This is disgusting.', 'I am afraid.', 'This is so exciting!', 'I am sad.', 'What a surprise!']
    expected_emotions = ['joy', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    for i, text in enumerate(sample_texts):
        result = detect_emotion(text)
        assert result[0]['label'].lower() == expected_emotions[i]

# call_test_function_code --------------------

test_detect_emotion()