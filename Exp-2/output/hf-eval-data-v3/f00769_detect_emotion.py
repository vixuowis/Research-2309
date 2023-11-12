# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_emotion(text):
    """
    Detects the emotion from the given text using a pre-trained model.

    Args:
        text (str): The text from which to detect the emotion.

    Returns:
        dict: The detected emotion and its score.

    Raises:
        OSError: If there is a problem with the model loading or prediction.
    """
    try:
        emotion_detector = pipeline('sentiment-analysis', model='michellejieli/emotion_text_classifier')
        user_emotion = emotion_detector(text)
        return user_emotion
    except OSError as e:
        print(f'Error: {e}')

# test_function_code --------------------

def test_detect_emotion():
    """
    Tests the detect_emotion function with some test cases.
    """
    assert detect_emotion('I love this!')[0]['label'] in ['anger', 'disgust', 'fear', 'joy', 'neutrality', 'sadness', 'surprise']
    assert detect_emotion('I hate this!')[0]['label'] in ['anger', 'disgust', 'fear', 'joy', 'neutrality', 'sadness', 'surprise']
    assert detect_emotion('This is scary!')[0]['label'] in ['anger', 'disgust', 'fear', 'joy', 'neutrality', 'sadness', 'surprise']
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_emotion()