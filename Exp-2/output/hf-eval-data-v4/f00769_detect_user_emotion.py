# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_user_emotion(text):
    """
    Detects the user's emotion from a given text response.

    Parameters:
    text (str): The response text from the user.

    Returns:
    dict: A dictionary containing the detected emotion and its confidence score.
    """
    # Initialize the sentiment analysis pipeline with the specific model
    emotion_detector = pipeline('sentiment-analysis', model='michellejieli/emotion_text_classifier')
    # Get the emotion prediction for the given text
    result = emotion_detector(text)
    # Return the emotion and its score
    return result[0]

# test_function_code --------------------

def test_detect_user_emotion():
    print("Testing detect_user_emotion function.")

    # Test case 1: Neutral response
    neutral_response = "I guess that's okay."
    assert 'neutrality' in detect_user_emotion(neutral_response)['label'], "Test case 1 failed: Expected 'neutrality'."

    # Test case 2: Positive response
    positive_response = "I'm so happy to hear that!"
    assert 'joy' in detect_user_emotion(positive_response)['label'], "Test case 2 failed: Expected 'joy'."

    # Test case 3: Negative response
    negative_response = "This is really upsetting."
    assert 'sadness' in detect_user_emotion(negative_response)['label'], "Test case 3 failed: Expected 'sadness'."

    print("All test cases passed!")

# Running the test function
test_detect_user_emotion()